import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.changeDataset import ChangeDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logger = get_logger()


# ======================= Grid Figure Helper (自包含，不再依赖外部文件) =======================

def _load_img(path, mode='RGB', size=None):
    """Load image and resize. mode: 'RGB' or 'L'."""
    img = Image.open(path).convert(mode)
    if size is not None:
        resample = Image.BICUBIC if mode == 'RGB' else Image.NEAREST
        img = img.resize(size, resample)
    return np.array(img)

def make_cd_grid(rows, out_path, tile_size=None, dpi=300):
    """
    rows: list[dict]
      {
        "dataset": "WHU-CD",
        "pre":  "/.../_tiles/pre/xxx.png",
        "post": "/.../_tiles/post/xxx.png",
        "gt":   "/.../_tiles/gt/xxx.png",
        "m1":   "/.../_overlay/xxx.png",   # 方法1 overlay
        "m2":   "/.../_overlay/xxx.png",   # 方法2 overlay
        "m3":   "/.../_overlay/xxx.png",   # Ours overlay
        "label1": "DDPM-CD",
        "label2": "ChangeMamba",
        "label3": "M-CD (Ours)"
      }
    """
    assert len(rows) > 0, "rows 不能为空"
    cols = 6
    # 统一 tile 尺寸：默认与第一行 GT 尺寸一致
    if tile_size is None:
        g0 = Image.open(rows[0]["gt"])
        tile_size = (g0.width, g0.height)

    fig_w = cols * tile_size[0] / dpi + 2.0
    fig_h = len(rows) * tile_size[1] / dpi + 1.5
    fig, axes = plt.subplots(len(rows), cols, figsize=(fig_w, fig_h), dpi=dpi)
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ["Pre-Change Image", "Post-Change Image", "Ground Truth", None, None, None]

    for i, r in enumerate(rows):
        pre  = _load_img(r["pre"],  "RGB", tile_size)
        post = _load_img(r["post"], "RGB", tile_size)
        gt   = _load_img(r["gt"],   "L",   tile_size)
        gt3  = np.stack([gt, gt, gt], axis=-1)

        o1 = _load_img(r["m1"], "RGB", tile_size)
        o2 = _load_img(r["m2"], "RGB", tile_size)
        o3 = _load_img(r["m3"], "RGB", tile_size)

        tiles = [pre, post, gt3, o1, o2, o3]
        for j in range(cols):
            ax = axes[i, j]
            ax.imshow(tiles[j])
            ax.set_axis_off()
            if j == 0:
                ax.set_ylabel(str(r["dataset"]), rotation=90, fontsize=10, labelpad=10, va="center")
            if i == 0:
                if j < 3:
                    ax.set_title(col_titles[j], fontsize=10)
                elif j == 3:
                    ax.set_title(str(r["label1"]), fontsize=10)
                elif j == 4:
                    ax.set_title(str(r["label2"]), fontsize=10)
                elif j == 5:
                    ax.set_title(str(r["label3"]), fontsize=10)

    legend_elems = [
        Patch(facecolor="black", edgecolor="black", label="TN"),
        Patch(facecolor="white", edgecolor="black", label="TP"),
        Patch(facecolor="red", edgecolor="black", label="FP"),
        Patch(facecolor="green", edgecolor="black", label="FN"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ================================== Evaluator ==================================

class SegEvaluator(Evaluator):
    def __init__(self, dataset, class_num, norm_mean, norm_std, network,
                 multi_scales, is_flip, devices, verbose=False,
                 save_path=None, show_image=False, config=None):
        super().__init__(dataset, class_num, norm_mean, norm_std, network,
                         multi_scales, is_flip, devices, verbose,
                         save_path, show_image, config)
        # 供 train.py 读取完整摘要
        self.last_metrics_summary = None

    def func_per_iteration(self, data, device, config):
        As = data['A']
        Bs = data['B']
        label = data['gt']
        name = data['fn']

        pred = self.sliding_eval_rgbX(As, Bs, config.eval_crop_size,
                                      config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path + '_color')
            ensure_dir(self.save_path + '_overlay')   # 叠加图目录
            ensure_dir(self.save_path + '_tiles')     # 统一尺寸tile
            fn = name + '.png'

            # 1) 保存灰度预测（0/1 -> 0/255）
            result_img = Image.fromarray((pred.astype(np.uint8) * 255), mode='L')
            result_img.save(os.path.join(self.save_path + '_color', fn))
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

            # 2) 读取 GT，生成红/绿/白/黑叠加图
            gt_path = os.path.join(self.dataset._root_path, 'gt', name + self.dataset._gt_format)
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)
            gt_bin = (gt_arr > 127).astype(np.uint8)

            pr_bin = (pred > 0).astype(np.uint8)
            tp = (gt_bin == 1) & (pr_bin == 1)
            fp = (gt_bin == 0) & (pr_bin == 1)
            fnm = (gt_bin == 1) & (pr_bin == 0)
            tn = (gt_bin == 0) & (pr_bin == 0)

            h, w = gt_bin.shape
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            overlay[tn]  = (0,   0,   0)     # TN 黑
            overlay[tp]  = (255, 255, 255)   # TP 白
            overlay[fp]  = (255, 0,   0)     # FP 红
            overlay[fnm] = (0,   255, 0)     # FN 绿
            Image.fromarray(overlay).save(os.path.join(self.save_path + '_overlay', fn))

            # 3) 保存统一尺寸的 pre/post/gt tiles（默认与 GT 同尺寸）
            A_path = os.path.join(self.dataset._root_path, 'A', name + self.dataset._A_format)
            B_path = os.path.join(self.dataset._root_path, 'B', name + self.dataset._B_format)

            def _load_rgb(path, target_size):
                img = Image.open(path).convert('RGB').resize(target_size, Image.BICUBIC)
                return img

            def _save_tile(img, subdir, fname):
                out_dir = os.path.join(self.save_path + '_tiles', subdir)
                ensure_dir(out_dir)
                img.save(os.path.join(out_dir, fname))

            target_size = (gt_img.width, gt_img.height)  # 与 GT 同尺寸；如需固定尺寸可改为(256,256)
            _save_tile(_load_rgb(A_path, target_size), 'pre',  fn)
            _save_tile(_load_rgb(B_path, target_size), 'post', fn)
            _save_tile(Image.fromarray((gt_bin * 255).astype(np.uint8)).convert('RGB'), 'gt', fn)

        if self.show_image:
            colors = self.dataset.get_class_colors()
            # 可按需增加可视化窗口

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, recall, precision, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
            hist, correct, labeled)
        result_line = print_iou(iou, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc,
                                self.dataset.class_names, show_no_back=False)

        # ==== pack summary for external logging ====
        f1_per_class = []
        try:
            for p, r in zip(precision, recall):
                denom = (p + r) if (p + r) != 0 else 1e-6
                f1_per_class.append(2 * p * r / denom)
            f1_macro = float(np.nanmean(np.array(f1_per_class)))
        except Exception:
            f1_macro = None
        self.last_metrics_summary = {
            'F1': f1_macro,
            'IoU': float(mean_IoU) if isinstance(mean_IoU, (int, float, np.floating)) else mean_IoU,
            'OA': float(pixel_acc) if isinstance(pixel_acc, (int, float, np.floating)) else pixel_acc
        }
        # ============================================
        return result_line, mean_IoU


# ================================== Main ==================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--dataset_name', '-n', default='mfnet', type=str)
    # 与原设定一致；默认 test
    parser.add_argument('--split', '-c', default='test', choices=['val', 'test'], type=str)

    # ==== 生成对比大图的可选参数 ====
    parser.add_argument('--make_grid', action='store_true',
                        help='评估结束后，拼论文对比大图')
    parser.add_argument('--grid_ddpm', type=str, default=None,
                        help='方法1(DDPM) 的评估输出根目录（包含 *_overlay 和 *_tiles）')
    parser.add_argument('--grid_cm', type=str, default=None,
                        help='方法2(ChangeMamba) 的评估输出根目录')
    parser.add_argument('--grid_ours', type=str, default=None,
                        help='方法3(Ours) 的评估输出根目录，默认用 --save_path')
    parser.add_argument('--grid_k', type=int, default=4,
                        help='拼图采样的样本数量（默认前 k 个文件名）')
    parser.add_argument('--grid_out', type=str, default=None,
                        help='拼图输出路径（默认保存到 ours 目录下 grid_<dataset>.png）')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    dataset_name = args.dataset_name
    if dataset_name == 'DSIFN':
        from configs.config_dsifn import config
    elif dataset_name == 'WHU':
        from configs.config_whu import config
    elif dataset_name == 'CDD':
        from configs.config_cdd import config
    elif dataset_name == 'LEVIR':
        from configs.config_levir import config
    else:
        raise ValueError('Not a valid dataset name')

    # 构建模型（上卡）
    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d).cuda(all_dev[0])

    # ===== 模型结构与参数摘要 =====
    print("\n" + "=" * 80)
    print("模型结构摘要:")
    print(network)

    print("\n详细参数分布:")
    total_params = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name:60} | {num_params:10,d} 参数 | 形状: {tuple(param.shape)}")

    print("-" * 80)
    print(f"总可训练参数: {total_params:,d}")
    print("=" * 80 + "\n")

    flops = network.flops()
    print("Gflops of the network: ", flops / (10 ** 9))
    print("number of paramters: ",
          sum(p.numel() if p.requires_grad else 0 for p in network.parameters()))

    # 数据集
    data_setting = {
        'root':        config.root_folder,
        'A_format':    config.A_format,
        'B_format':    config.B_format,
        'gt_format':   config.gt_format,
        'class_names': config.class_names
    }
    val_pre = ValPre()
    dataset = ChangeDataset(data_setting, args.split, val_pre)

    # 日志目录：只为父目录建目录，避免把 .log 当目录
    log_dir = os.path.dirname(config.val_log_file)
    link_log_dir = os.path.dirname(config.link_val_log_file)
    if log_dir:
        ensure_dir(log_dir)
    if link_log_dir:
        ensure_dir(link_log_dir)

    segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                             config.norm_std, network,
                             config.eval_scale_array, config.eval_flip,
                             all_dev, args.verbose, args.save_path,
                             args.show_image, config)

    _, mean_IoU = segmentor.run_eval(config.checkpoint_dir, args.epochs,
                                     config.val_log_file, config.link_val_log_file)

    # ===================== 评估后生成论文对比大图（可选） =====================
    if args.make_grid:
        try:
            # ours 目录默认取本次 --save_path
            ours_base = args.grid_ours if args.grid_ours is not None else (args.save_path or "runs/vis_ours")
            ddpm_base = args.grid_ddpm or "runs/vis_ddpm"
            cm_base   = args.grid_cm   or "runs/vis_changemamba"

            # 取前 k 个样本（可改为你固定想展示的文件名列表）
            sample_names = dataset._file_names[:max(1, args.grid_k)]

            rows = []
            for fname in sample_names:
                fn = fname + '.png'
                rows.append({
                    "dataset": args.dataset_name,
                    "pre":  os.path.join(ours_base + "_tiles", "pre",  fn),
                    "post": os.path.join(ours_base + "_tiles", "post", fn),
                    "gt":   os.path.join(ours_base + "_tiles", "gt",   fn),
                    "m1":   os.path.join(ddpm_base + "_overlay", fn),
                    "m2":   os.path.join(cm_base   + "_overlay", fn),
                    "m3":   os.path.join(ours_base + "_overlay", fn),
                    "label1": "DDPM-CD",
                    "label2": "ChangeMamba",
                    "label3": "M-CD (Ours)"
                })

            out_fig = args.grid_out or os.path.join(ours_base, f"grid_{args.dataset_name}.png")
            ensure_dir(os.path.dirname(out_fig) or ".")
            make_cd_grid(rows, out_fig, tile_size=None, dpi=300)
            print(f"[Grid] Saved figure to: {out_fig}")
        except Exception as e:
            print(f"[Grid] Skip making grid: {e}")
