import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# ===== 手动指定8张图片的路径和标签 =====
# 格式： (图片路径, 真实标签, 预测标签)
fp_samples = [
    ('Normal097.jpg', 'normal', 'initial_cold'),
    ('Normal039.jpg', 'normal', 'initial_cold')
]

fn_samples = [
    ('Initial-cold-stresss088.jpg', 'initial_cold', 'normal'),
    ('Initial-cold-stresss019.jpg', 'initial_cold', 'normal')
]

# ===== 绘制图像 =====
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 上半部分：假阳性
for i, (img_path, true, pred) in enumerate(fp_samples):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 可选：resize到统一尺寸以美观
    img = cv2.resize(img, (224, 224))
    axes[0, i].imshow(img)
    axes[0, i].set_title(f'FP: true={true}\npred={pred}', fontsize=10)
    axes[0, i].axis('off')

# 下半部分：假阴性
for i, (img_path, true, pred) in enumerate(fn_samples):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    axes[1, i].imshow(img)
    axes[1, i].set_title(f'FN: true={true}\npred={pred}', fontsize=10)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('figure11_error_analysis.png', dpi=300)
plt.show()