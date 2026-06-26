from ultralytics import YOLO
import os
import random
import numpy as np
import torch
import gc
import traceback
import time

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    """
    设置随机种子确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ 随机种子 {seed} 已设置完成")


def cleanup_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    torch.cuda.empty_cache()


def train_with_seed(seed, model_path, project_name, exp_name):
    """
    使用指定随机种子训练模型
    """
    try:
        set_seed(seed)

        print(f"加载模型: {model_path}")
        model = YOLO(model=model_path)

        # 加载预训练权重（如果可用）
        pretrained_path = 'yolov10n.pt'
        if os.path.exists(pretrained_path):
            try:
                model.load(pretrained_path)
                print(f"✓ 加载预训练权重: {pretrained_path}")
            except Exception as e:
                print(f"⚠ 预训练权重加载失败: {e}, 从头开始训练")
        else:
            print("ℹ 未找到预训练权重，从头开始训练")

        train_args = {
            'data': './data.yaml',
            'epochs': 200,
            'batch': 8,
            'device': '0',
            'imgsz': 640,
            'workers': 2,
            'cache': False,
            'amp': True,
            'mosaic': False,
            'project': project_name,
            'name': exp_name,
            'seed': seed,
            'patience': 50,
            'save_period': 10,
            'exist_ok': True,
            'verbose': True,
        }

        print(f"\n{'=' * 50}")
        print(f"🚀 开始训练 - 随机种子: {seed}")
        print(f"📁 实验名称: {exp_name}")
        print(f"{'=' * 50}")

        results = model.train(**train_args)
        cleanup_memory()
        return results

    except Exception as e:
        print(f"❌ 训练失败 - 随机种子 {seed}")
        print(f"错误信息: {str(e)}")
        traceback.print_exc()
        cleanup_memory()
        return None


def extract_metrics(result):
    """
    从验证结果中提取 mAP50, mAP50-95, 召回率, F1 分数
    兼容不同版本的 Ultralytics YOLO
    """
    # 默认值
    map50 = map50_95 = recall = precision = f1 = 0.0

    if hasattr(result, 'box'):
        box = result.box
        # 尝试从 box 属性直接获取
        map50 = getattr(box, 'map50', None)
        map50_95 = getattr(box, 'map', None)          # mAP@50-95 通常存储在 map 属性中
        recall = getattr(box, 'recall', None)
        precision = getattr(box, 'precision', None)   # 部分版本可能提供 precision
        f1 = getattr(box, 'f1', None)

    # 如果某些指标缺失，尝试从 results_dict 获取（与 CSV 列名一致）
    if (map50 is None or map50_95 is None or recall is None or
        precision is None or f1 is None) and hasattr(result, 'results'):
        rd = result.results_dict
        map50 = rd.get('metrics/mAP50(B)', map50 or 0.0)
        map50_95 = rd.get('metrics/mAP50-95(B)', map50_95 or 0.0)
        recall = rd.get('metrics/recall(B)', recall or 0.0)
        precision = rd.get('metrics/precision(B)', precision or 0.0)
        # 如果 f1 仍未获取，则计算
        if f1 is None:
            f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

    # 如果仍然缺失，赋值为 0
    map50 = float(map50) if map50 is not None else 0.0
    map50_95 = float(map50_95) if map50_95 is not None else 0.0
    recall = float(recall) if recall is not None else 0.0
    f1 = float(f1) if f1 is not None else 0.0

    return {
        'map50': map50,
        'map50_95': map50_95,
        'recall': recall,
        'f1': f1
    }


def validate_model(seed, project_name):
    """验证指定种子的模型（使用验证集）"""
    model_file = f'{project_name}/exp_seed_{seed}/weights/best.pt'

    if not os.path.exists(model_file):
        print(f"⚠ 模型文件不存在: {model_file}")
        return None

    try:
        print(f"验证模型: {model_file}")
        val_model = YOLO(model_file)
        val_results = val_model.val(
            data='./data.yaml',
            split='val',          # 使用验证集（如果没有 test 集）
            imgsz=640,
            device='0',
            verbose=True
        )
        return val_results
    except Exception as e:
        print(f"❌ 验证失败 - 随机种子 {seed}: {e}")
        return None


if __name__ == '__main__':
    # 定义要使用的随机种子
    seeds = [42, 3407, 2026]

    # 定义模型路径和项目名称
    model_path = 'ultralytics/cfg/models/v10/yolov10n.yaml'
    project_name = 'runs/train_seeds1'

    os.makedirs(project_name, exist_ok=True)

    successful_trains = []
    all_results = {}

    print(f"🔧 开始多随机种子实验")
    print(f"📊 随机种子列表: {seeds}")
    print(f"📂 输出目录: {project_name}")
    print(f"{'=' * 50}\n")

    # 为每个种子训练一个模型
    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 50}")
        print(f"📈 进度: {i + 1}/{len(seeds)}")
        print(f"🌱 当前随机种子: {seed}")
        print(f"{'=' * 50}")

        exp_dir = f'{project_name}/exp_seed_{seed}'
        if os.path.exists(exp_dir):
            print(f"ℹ 实验目录已存在: {exp_dir}")
            print("是否覆盖？(y/n): ", end="")
            choice = input().strip().lower()
            if choice != 'y':
                print(f"跳过随机种子 {seed} 的训练")
                if os.path.exists(f'{exp_dir}/weights/best.pt'):
                    successful_trains.append(seed)
                continue

        start_time = time.time()
        exp_name = f'exp_seed_{seed}'

        results = train_with_seed(seed, model_path, project_name, exp_name)

        if results is not None:
            elapsed_time = time.time() - start_time
            successful_trains.append(seed)
            all_results[seed] = results

            print(f"\n✅ 随机种子 {seed} 训练完成")
            print(f"⏱️ 训练时间: {elapsed_time:.1f} 秒")
            if hasattr(results, 'best_fitness'):
                print(f"🏆 最佳mAP: {results.best_fitness:.4f}")
            if hasattr(results, 'save_dir'):
                print(f"💾 保存路径: {results.save_dir}")
        else:
            print(f"❌ 随机种子 {seed} 训练失败")

    print(f"\n{'=' * 50}")
    print(f"🎯 训练完成总结")
    print(f"{'=' * 50}")
    print(f"✅ 成功训练: {len(successful_trains)}/{len(seeds)}")
    print(f"📊 成功种子: {successful_trains}")

    # 验证所有成功训练的模型
    if successful_trains:
        print(f"\n{'=' * 50}")
        print(f"🔍 开始模型验证")
        print(f"{'=' * 50}")

        val_results = {}
        for seed in successful_trains:
            print(f"\n验证随机种子 {seed}...")
            result = validate_model(seed, project_name)
            if result is not None:
                val_results[seed] = result

        # 打印验证结果摘要
        if val_results:
            print(f"\n{'=' * 50}")
            print(f"📋 验证结果摘要")
            print(f"{'=' * 50}")

            # 表头
            print(f"{'种子':<8} {'mAP@50':<10} {'mAP@50-95':<12} {'召回率':<10} {'F1分数':<10}")
            print("-" * 60)

            # 收集所有指标用于后续统计
            metrics_list = []

            for seed, result in val_results.items():
                metrics = extract_metrics(result)
                metrics_list.append(metrics)
                print(f"{seed:<8} {metrics['map50']:<10.4f} {metrics['map50_95']:<12.4f} "
                      f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")

            # 计算统计信息
            if metrics_list:
                map50_vals = [m['map50'] for m in metrics_list]
                map95_vals = [m['map50_95'] for m in metrics_list]
                recall_vals = [m['recall'] for m in metrics_list]
                f1_vals = [m['f1'] for m in metrics_list]

                print(f"\n📈 统计信息:")
                print(f"mAP@50   - 平均值: {np.mean(map50_vals):.4f} 标准差: {np.std(map50_vals):.4f}  "
                      f"范围: [{np.min(map50_vals):.4f}, {np.max(map50_vals):.4f}]")
                print(f"mAP@50-95- 平均值: {np.mean(map95_vals):.4f} 标准差: {np.std(map95_vals):.4f}")
                print(f"召回率    - 平均值: {np.mean(recall_vals):.4f} 标准差: {np.std(recall_vals):.4f}")
                print(f"F1分数   - 平均值: {np.mean(f1_vals):.4f} 标准差: {np.std(f1_vals):.4f}")

                # 保存统计结果到文件
                stats_file = f"{project_name}/validation_stats.txt"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    f.write("多随机种子验证结果统计\n")
                    f.write("=" * 60 + "\n")
                    for i, seed in enumerate(val_results.keys()):
                        m = metrics_list[i]
                        f.write(f"种子 {seed}: mAP@50={m['map50']:.4f}, mAP@50-95={m['map50_95']:.4f}, "
                                f"召回率={m['recall']:.4f}, F1={m['f1']:.4f}\n")
                    f.write("\n统计信息:\n")
                    f.write(f"mAP@50   - 平均值: {np.mean(map50_vals):.4f} 标准差: {np.std(map50_vals):.4f}  "
                            f"范围: [{np.min(map50_vals):.4f}, {np.max(map50_vals):.4f}]\n")
                    f.write(f"mAP@50-95- 平均值: {np.mean(map95_vals):.4f} 标准差: {np.std(map95_vals):.4f}\n")
                    f.write(f"召回率    - 平均值: {np.mean(recall_vals):.4f} 标准差: {np.std(recall_vals):.4f}\n")
                    f.write(f"F1分数   - 平均值: {np.mean(f1_vals):.4f} 标准差: {np.std(f1_vals):.4f}\n")

                print(f"\n💾 统计结果已保存到: {stats_file}")
        else:
            print("\n⚠ 没有成功验证的模型，跳过统计")
    else:
        print("\n⚠ 没有成功训练的模型，跳过验证")

    print(f"\n{'=' * 50}")
    print(f"🎉 所有任务完成！")
    print(f"{'=' * 50}")