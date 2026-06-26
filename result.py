import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # 解决 OpenMP 冲突

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ultralytics import YOLO

# ==================== 配置区域 ====================
seeds = [42, 3407, 2026]  # 随机种子列表
project_dir = "runs/YOLOv11-DySample-slide"  # 训练结果根目录
data_yaml = "./data.yaml"  # 数据集配置文件
save_dir = "./seed_summary/YOLOv11-DySample-slide"  # 汇总结果保存目录
os.makedirs(save_dir, exist_ok=True)


# ==================== 辅助函数 ====================
def extract_class_metrics(val_results):
    """从验证结果中提取每个类别的 AP50, AP50-95, Recall, F1-score."""
    if not hasattr(val_results, "box") or not hasattr(val_results.box, "ap_class_index"):
        raise ValueError("验证结果中未找到类别指标")

    class_names = (
        val_results.names
        if hasattr(val_results, "names")
        else {i: str(i) for i in range(len(val_results.box.ap_class_index))}
    )
    class_ids = val_results.box.ap_class_index
    ap50 = val_results.box.ap50
    ap50_95 = val_results.box.ap
    precision = val_results.box.p
    recall = val_results.box.r
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

    metrics_dict = {}
    for i, cls_id in enumerate(class_ids):
        cls_name = class_names.get(cls_id, str(cls_id))
        metrics_dict[cls_name] = {"map50": ap50[i], "map50_95": ap50_95[i], "recall": recall[i], "f1": f1[i]}
    return metrics_dict


def extract_pr_curves(val_results):
    """从验证结果中提取每个类别的 PR 曲线数据点（修复类型和维度错误）."""
    pr_data = {}
    if not hasattr(val_results.box, "curves_results"):
        return pr_data
    class_names = val_results.names if hasattr(val_results, "names") else {}
    for res in val_results.box.curves_results:
        # res 格式: (precision, recall, ap, class_id)
        precision, recall, ap, cls_id = res
        # 转换为 numpy 数组并展平为一维
        precision = np.array(precision).flatten()
        recall = np.array(recall).flatten()
        # 检查长度是否一致
        if len(precision) != len(recall):
            print(f"警告: 类别 {cls_id} 的 PR 曲线数据长度不一致，跳过该曲线")
            continue
        # 将 ap 转换为浮点数
        try:
            ap = float(ap)
        except (TypeError, ValueError):
            ap = 0.0
        cls_name = class_names.get(cls_id, str(cls_id))
        pr_data[cls_name] = {"precision": precision, "recall": recall, "ap": ap}
    return pr_data


def extract_confusion_matrix(val_results):
    """从验证结果中提取归一化混淆矩阵."""
    if hasattr(val_results, "confusion_matrix") and hasattr(val_results.confusion_matrix, "matrix"):
        cm = val_results.confusion_matrix.matrix
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-16)
        return cm_norm
    return None


def plot_average_loss_accuracy_curves(seeds, project_dir, save_path):
    """绘制平均后的损失和精度曲线（从 results.csv 计算均值和标准差）."""
    dfs = []
    for seed in seeds:
        csv_path = os.path.join(project_dir, f"exp_seed_{seed}", "results.csv")
        if not os.path.exists(csv_path):
            print(f"警告: {csv_path} 不存在，跳过")
            continue
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        dfs.append(df)

    if not dfs:
        print("没有可用的 results.csv 文件")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    epochs = sorted(all_df["epoch"].unique())
    metrics = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    avg_data = {"epoch": epochs}
    std_data = {}
    for m in metrics:
        avg_vals, std_vals = [], []
        for e in epochs:
            vals = all_df[all_df["epoch"] == e][m].values
            avg_vals.append(np.mean(vals))
            std_vals.append(np.std(vals))
        avg_data[m] = avg_vals
        std_data[m] = std_vals

    plt.figure(figsize=(16, 10))

    # 训练损失
    plt.subplot(2, 4, 1)
    for m in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
        plt.plot(epochs, avg_data[m], label=m.split("/")[-1].replace("_loss", ""))
        plt.fill_between(
            epochs,
            np.array(avg_data[m]) - np.array(std_data[m]),
            np.array(avg_data[m]) + np.array(std_data[m]),
            alpha=0.2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Avg ± Std)")
    plt.legend()
    plt.grid(True)

    # 验证损失
    plt.subplot(2, 4, 2)
    for m in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
        plt.plot(epochs, avg_data[m], label=m.split("/")[-1].replace("_loss", ""))
        plt.fill_between(
            epochs,
            np.array(avg_data[m]) - np.array(std_data[m]),
            np.array(avg_data[m]) + np.array(std_data[m]),
            alpha=0.2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss (Avg ± Std)")
    plt.legend()
    plt.grid(True)

    # 精确率
    plt.subplot(2, 4, 3)
    m = "metrics/precision(B)"
    plt.plot(epochs, avg_data[m], "b-", label="Precision")
    plt.fill_between(
        epochs,
        np.array(avg_data[m]) - np.array(std_data[m]),
        np.array(avg_data[m]) + np.array(std_data[m]),
        alpha=0.2,
        color="b",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision (Avg ± Std)")
    plt.grid(True)

    # 召回率
    plt.subplot(2, 4, 4)
    m = "metrics/recall(B)"
    plt.plot(epochs, avg_data[m], "g-", label="Recall")
    plt.fill_between(
        epochs,
        np.array(avg_data[m]) - np.array(std_data[m]),
        np.array(avg_data[m]) + np.array(std_data[m]),
        alpha=0.2,
        color="g",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall (Avg ± Std)")
    plt.grid(True)

    # mAP50
    plt.subplot(2, 4, 5)
    m = "metrics/mAP50(B)"
    plt.plot(epochs, avg_data[m], "r-", label="mAP50")
    plt.fill_between(
        epochs,
        np.array(avg_data[m]) - np.array(std_data[m]),
        np.array(avg_data[m]) + np.array(std_data[m]),
        alpha=0.2,
        color="r",
    )
    plt.xlabel("Epoch")
    plt.ylabel("mAP50")
    plt.title("mAP@50 (Avg ± Std)")
    plt.grid(True)

    # mAP50-95
    plt.subplot(2, 4, 6)
    m = "metrics/mAP50-95(B)"
    plt.plot(epochs, avg_data[m], "m-", label="mAP50-95")
    plt.fill_between(
        epochs,
        np.array(avg_data[m]) - np.array(std_data[m]),
        np.array(avg_data[m]) + np.array(std_data[m]),
        alpha=0.2,
        color="m",
    )
    plt.xlabel("Epoch")
    plt.ylabel("mAP50-95")
    plt.title("mAP@50-95 (Avg ± Std)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"平均损失精度曲线已保存至: {save_path}")


def plot_average_pr_curves(pr_data_list, seeds, class_names_dict, save_dir):
    """为每个类别绘制平均PR曲线（叠加三个种子的曲线）."""
    all_classes = set()
    for pr in pr_data_list:
        all_classes.update(pr.keys())
    all_classes = sorted(all_classes)

    for cls in all_classes:
        plt.figure(figsize=(8, 6))
        aps = []
        for idx, (seed, pr) in enumerate(zip(seeds, pr_data_list)):
            if cls in pr:
                prec = pr[cls]["precision"]
                rec = pr[cls]["recall"]
                ap = pr[cls]["ap"]
                # 确保 ap 是浮点数
                ap = float(ap)
                aps.append(ap)
                plt.plot(rec, prec, label=f"Seed {seed} (AP={ap:.3f})", linewidth=1.5)
        if aps:
            avg_ap = np.mean(aps)
            plt.title(f"PR Curve - {cls} (Avg AP={avg_ap:.3f})")
        else:
            plt.title(f"PR Curve - {cls}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(os.path.join(save_dir, f"PR_{cls}.png"), dpi=300)
        plt.close()
    print(f"PR曲线已保存至: {save_dir}")


def plot_average_confusion_matrix(cm_list, class_names_dict, save_path):
    """绘制平均归一化混淆矩阵热力图."""
    if not cm_list:
        return
    avg_cm = np.mean(cm_list, axis=0)
    n_classes = avg_cm.shape[0]
    class_names_list = [class_names_dict.get(i, f"class_{i}") for i in range(n_classes)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names_list, yticklabels=class_names_list)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Average Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"平均混淆矩阵已保存至: {save_path}")


# ==================== 主程序 ====================
def main():
    val_results_list = []
    pr_data_list = []
    cm_list = []
    all_class_metrics = []

    # 获取第一个有效的类别名称字典（用于后续绘图）
    class_names_dict = None

    for seed in seeds:
        model_path = os.path.join(project_dir, f"exp_seed_{seed}", "weights", "best.pt")
        if not os.path.exists(model_path):
            print(f"警告: 种子 {seed} 的模型文件不存在: {model_path}，跳过")
            continue

        print(f"正在验证种子 {seed} ...")
        try:
            model = YOLO(model_path)
            val_res = model.val(data=data_yaml, split="val", imgsz=640, device="0", verbose=True)
        except Exception as e:
            print(f"种子 {seed} 验证失败: {e}")
            continue

        val_results_list.append(val_res)

        # 提取各类指标
        try:
            all_class_metrics.append(extract_class_metrics(val_res))
        except Exception as e:
            print(f"提取类别指标失败 (种子 {seed}): {e}")
            continue

        pr_data_list.append(extract_pr_curves(val_res))
        cm = extract_confusion_matrix(val_res)
        if cm is not None:
            cm_list.append(cm)

        # 从第一个有效的验证结果中获取类别名称字典
        if class_names_dict is None and hasattr(val_res, "names"):
            class_names_dict = val_res.names

    if not val_results_list:
        print("没有成功的验证结果，程序退出")
        return

    # 如果没有获取到类别名称，创建一个默认的
    if class_names_dict is None:
        # 尝试从第一个类别指标中推断类别数
        if all_class_metrics:
            n_classes = len(all_class_metrics[0])
            class_names_dict = {i: f"class_{i}" for i in range(n_classes)}
        else:
            class_names_dict = {0: "unknown"}

    # 1. 输出各类别平均指标表格
    all_classes = sorted(set().union(*[cm.keys() for cm in all_class_metrics]))
    rows = []
    for cls in all_classes:
        map50 = [cm.get(cls, {}).get("map50", np.nan) for cm in all_class_metrics]
        map95 = [cm.get(cls, {}).get("map50_95", np.nan) for cm in all_class_metrics]
        rec = [cm.get(cls, {}).get("recall", np.nan) for cm in all_class_metrics]
        f1 = [cm.get(cls, {}).get("f1", np.nan) for cm in all_class_metrics]

        # 去除NaN
        map50 = [v for v in map50 if not np.isnan(v)]
        map95 = [v for v in map95 if not np.isnan(v)]
        rec = [v for v in rec if not np.isnan(v)]
        f1 = [v for v in f1 if not np.isnan(v)]

        if map50:
            rows.append(
                {
                    "Class": cls,
                    "mAP50_mean": np.mean(map50),
                    "mAP50_std": np.std(map50),
                    "mAP50-95_mean": np.mean(map95),
                    "mAP50-95_std": np.std(map95),
                    "Recall_mean": np.mean(rec),
                    "Recall_std": np.std(rec),
                    "F1_mean": np.mean(f1),
                    "F1_std": np.std(f1),
                }
            )

    df = pd.DataFrame(rows)
    print("\n==================== 各类别平均指标 ====================")
    print(df.to_string(index=False, float_format="%.4f"))
    csv_path = os.path.join(save_dir, "class_metrics_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"指标汇总已保存至: {csv_path}")

    # 2. 平均损失精度曲线
    plot_average_loss_accuracy_curves(seeds, project_dir, os.path.join(save_dir, "average_loss_accuracy_curves.png"))

    # 3. PR曲线
    if pr_data_list and any(pr_data_list):
        plot_average_pr_curves(pr_data_list, seeds, class_names_dict, save_dir)
    else:
        print("跳过PR曲线（无数据）")

    # 4. 平均混淆矩阵
    if cm_list:
        plot_average_confusion_matrix(cm_list, class_names_dict, os.path.join(save_dir, "average_confusion_matrix.png"))
    else:
        print("跳过混淆矩阵（无数据）")

    print(f"\n所有图表已保存至: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()
