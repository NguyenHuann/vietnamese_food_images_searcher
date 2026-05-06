import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# 1. TẢI DỮ LIỆU VECTOR

print("Đang tải dữ liệu vector...")
with open("../models/vectors.pkl", "rb") as f:
    vectors = pickle.load(f)
with open("../models/paths.pkl", "rb") as f:
    paths = pickle.load(f)

# Lấy nhãn thông minh từ đường dẫn
labels = [os.path.basename(os.path.dirname(p)) for p in paths]

# Chia tập Cơ sở dữ liệu (80%) và tập Truy vấn (20%)
db_vectors, query_vectors, db_labels, query_labels_true = train_test_split(
    vectors, labels, test_size=0.2, random_state=42, stratify=labels
)

# 2. TÍNH TOÁN AVERAGE PRECISION (AP) CHO TỪNG TRUY VẤN
classes = sorted(list(set(labels)))
ap_scores_per_class = {c: [] for c in classes}

print("Đang chạy mô phỏng tìm kiếm và tính toán mAP...")
for q_vec, q_label in zip(query_vectors, query_labels_true):
    # Tính độ tương đồng với toàn bộ database
    sims = np.dot(db_vectors, q_vec)

    # Tạo nhãn nhị phân: 1 nếu ảnh trong DB cùng món với ảnh Query, ngược lại là 0
    y_true_binary = np.array([1 if db_label == q_label else 0 for db_label in db_labels])

    # Tính Average Precision (AP) cho riêng bức ảnh truy vấn này
    if np.sum(y_true_binary) > 0:
        ap = average_precision_score(y_true_binary, sims)
        ap_scores_per_class[q_label].append(ap)

# 3. TÍNH MEAN AVERAGE PRECISION (mAP) THEO LỚP
map_per_class = {}
for c in classes:
    if len(ap_scores_per_class[c]) > 0:
        map_per_class[c] = np.mean(ap_scores_per_class[c])
    else:
        map_per_class[c] = 0.0

# Tính mAP toàn hệ thống
overall_map = np.mean(list(map_per_class.values()))
print(f"\nCHỈ SỐ mAP TOÀN HỆ THỐNG: {overall_map * 100:.2f}%")

# 4. CHUẨN BỊ DỮ LIỆU ĐỂ VẼ BIỂU ĐỒ
# Sắp xếp các class theo điểm mAP từ cao xuống thấp
sorted_classes = sorted(map_per_class.keys(), key=lambda k: map_per_class[k], reverse=True)
sorted_maps = [map_per_class[k] * 100 for k in sorted_classes]  # Đổi sang phần trăm %

# 5. VẼ BIỂU ĐỒ BAR CHART (NẰM NGANG)
plt.figure(figsize=(14, 12))

# Dùng Seaborn để vẽ biểu đồ cột ngang, màu sắc thay đổi theo giá trị (palette 'viridis')
barplot = sns.barplot(
    x=sorted_maps,
    y=sorted_classes,
    palette="viridis",
    orient='h'
)

# Thêm con số % cụ thể ở đuôi mỗi cột
for i, val in enumerate(sorted_maps):
    barplot.text(val + 0.5, i, f'{val:.1f}%', color='black', va="center", fontsize=10)

# Căn chỉnh thẩm mỹ
plt.title(f'Đánh giá mAP (Mean Average Precision) của 30 món ăn\nmAP Toàn hệ thống: {overall_map * 100:.2f}%',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Điểm mAP (%)', fontsize=12)
plt.ylabel('Tên món ăn', fontsize=12)

# Giới hạn trục X từ 0 đến 100% để hiển thị chuẩn tỷ lệ
plt.xlim(0, 105)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Lưu biểu đồ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(BASE_DIR, "figures")
plot_path = os.path.join(out_dir, "map_per_class_chart.png")
plt.savefig(plot_path, dpi=300)
print(f"Đã lưu biểu đồ tại: {plot_path}")