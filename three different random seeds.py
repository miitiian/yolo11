import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ====================== 从截图手动输入三个种子的最终验证指标 ======================
# 数据来源：您提供的三张截图（种子42、3407、2026）
seed_data = {
    42: {
        'mAP50': 88.6,        # 截图 all 行 mAP50=0.886 → 88.6%
        'mAP5095': 66.8,       # 0.668 → 66.8%
        'Recall': 82.9,        # 0.829 → 82.9%
        'Precision': 86.1       # 0.861 → 86.1%
    },
    3407: {
        'mAP50': 89.5,
        'mAP5095': 66.9,
        'Recall': 83.4,
        'Precision': 87.5
    },
    2026: {
        'mAP50': 89.0,
        'mAP5095': 67.0,
        'Recall': 84.4,
        'Precision': 88.4
    }
}

# 计算每个种子的 F1-Score（百分比）
for seed in seed_data:
    P = seed_data[seed]['Precision'] / 100  # 转换为小数
    R = seed_data[seed]['Recall'] / 100
    f1 = 2 * P * R / (P + R)
    seed_data[seed]['F1'] = f1 * 100  # 存储为百分比

# ====================== 绘制柱状图 ======================
seeds = list(seed_data.keys())
metrics_names = ['mAP@0.5', 'Recall', 'F1-Score']
x = np.arange(len(seeds))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

for i, metric in enumerate(metrics_names):
    if metric == 'mAP@0.5':
        values = [seed_data[s]['mAP50'] for s in seeds]
    elif metric == 'Recall':
        values = [seed_data[s]['Recall'] for s in seeds]
    elif metric == 'F1-Score':
        values = [seed_data[s]['F1'] for s in seeds]
    bars = ax.bar(x + i*width, values, width, label=metric)
    # 在最优种子（3407）的柱子上添加星号标记
    if metric in ['mAP@0.5', 'Recall', 'F1-Score']:
        max_val_idx = np.argmax(values)
        if seeds[max_val_idx] == 3407:
            bar = bars[max_val_idx]
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, '*',
                    ha='center', va='bottom', fontsize=14, color='red')

ax.set_xlabel('Random Seed')
ax.set_ylabel('Score (%)')
ax.set_title('Performance Comparison under Different Random Seeds\n(Based on best.pt validation results)')
ax.set_xticks(x + width)
ax.set_xticklabels(seeds)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('bar_chart_best.png', dpi=300)
print("柱状图已保存为 bar_chart_best.png")

# ====================== 绘制折线图（使用 results.csv 的训练曲线，添加95%置信区间） ======================
# 请确保以下路径指向您的三个 results.csv 文件
seed_files = {
    42: 'runs/YOLOv11-DySample-slide/exp_seed_42/results.csv',
    3407: 'runs/YOLOv11-DySample-slide/exp_seed_3407/results.csv',
    2026: 'runs/YOLOv11-DySample-slide/exp_seed_2026/results.csv'
}

line_data = {}
for seed, file in seed_files.items():
    df = pd.read_csv(file)
    line_data[seed] = df[['epoch', 'metrics/mAP50(B)']].copy()
    line_data[seed]['metrics/mAP50(B)'] *= 100  # 转为百分比

# 合并数据
df_merged = pd.DataFrame()
for seed in seeds:
    df_seed = line_data[seed].set_index('epoch')['metrics/mAP50(B)'].rename(seed)
    df_merged = pd.concat([df_merged, df_seed], axis=1)

# 计算均值、标准差和95%置信区间
mean = df_merged.mean(axis=1)
std = df_merged.std(axis=1, ddof=1)
n = len(seeds)
t_value = stats.t.ppf(0.975, df=n-1)
ci = t_value * std / np.sqrt(n)

plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for seed, color in zip(seeds, colors):
    plt.plot(df_merged.index, df_merged[seed], color=color, linewidth=1.0, alpha=0.4, label=f'Seed {seed}')

plt.plot(df_merged.index, mean, color='red', linewidth=2.5, label='Mean')
plt.fill_between(df_merged.index, mean - ci, mean + ci, color='red', alpha=0.2, label='95% CI')

plt.xlabel('Epoch')
plt.ylabel('Test mAP@0.5 (%)')
plt.title('Test mAP@0.5 vs. Epoch with 95% Confidence Interval\n(Training curves from results.csv)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('line_chart.png', dpi=300)
print("折线图已保存为 line_chart.png")

# ====================== 打印三个种子的最终性能 ======================
print("\n===== 三个种子的最终测试性能 (基于 best.pt) =====")
for seed in seeds:
    print(f"Seed {seed}: mAP@0.5={seed_data[seed]['mAP50']:.1f}%, "
          f"mAP@0.5:0.95={seed_data[seed]['mAP5095']:.1f}%, "
          f"Recall={seed_data[seed]['Recall']:.1f}%, "
          f"F1={seed_data[seed]['F1']:.1f}%")

# 均值与标准差（基于 best.pt 结果）
mAP50_vals = [seed_data[s]['mAP50'] for s in seeds]
mAP5095_vals = [seed_data[s]['mAP5095'] for s in seeds]
recall_vals = [seed_data[s]['Recall'] for s in seeds]
f1_vals = [seed_data[s]['F1'] for s in seeds]

print("\n===== 三个种子的均值与标准差 (基于 best.pt) =====")
print(f"mAP@0.5:     {np.mean(mAP50_vals):.1f} ± {np.std(mAP50_vals):.1f}")
print(f"mAP@0.5:0.95: {np.mean(mAP5095_vals):.1f} ± {np.std(mAP5095_vals):.1f}")
print(f"Recall:       {np.mean(recall_vals):.1f} ± {np.std(recall_vals):.1f}")
print(f"F1-Score:     {np.mean(f1_vals):.1f} ± {np.std(f1_vals):.1f}")