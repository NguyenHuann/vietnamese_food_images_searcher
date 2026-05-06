import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

# 1. TẢI DỮ LIỆU VECTOR
print("Đang tải dữ liệu vector...")
with open("../models/vectors.pkl", "rb") as f:
    vectors = pickle.load(f)
with open("../models/paths.pkl", "rb") as f:
    paths = pickle.load(f)

# 2. XỬ LÝ NHÃN CHUẨN XÁC & CHIA TẬP TEST TỰ ĐỘNG
# Bất kể đường dẫn là "pho/1.jpg" hay "dataset/train/pho/1.jpg"
# Hàm này sẽ luôn lấy đúng thư mục chứa ảnh (tên món ăn)
labels = [os.path.basename(os.path.dirname(p)) for p in paths]

# Dùng train_test_split chia 80% DB (Train) và 20% Query (Val)
# stratify=labels giúp đảm bảo món nào cũng có 20% ảnh mang đi test
db_vectors, query_vectors, db_labels, query_labels_true = train_test_split(
    vectors, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nĐã chia dữ liệu thành công!")
print(f"   - Cơ sở dữ liệu (Database): {len(db_vectors)} ảnh")
print(f"   - Tập truy vấn kiểm thử (Query): {len(query_vectors)} ảnh\n")

# 3. MÔ PHỎNG QUÁ TRÌNH BẦU CHỌN k-NN
K = 5
query_labels_pred = []

print(f"Đang chạy kiểm thử k-NN với K={K}...")
for q_vec in query_vectors:
    # Tính độ tương đồng bằng Tích vô hướng (Dot Product)
    sims = np.dot(db_vectors, q_vec)

    # Lấy top K ảnh giống nhất
    top_k_indices = np.argsort(sims)[::-1][:K]
    top_k_labels = [db_labels[idx] for idx in top_k_indices]

    # Bầu chọn Majority Voting
    majority_label = Counter(top_k_labels).most_common(1)[0][0]
    query_labels_pred.append(majority_label)

# 4. TÍNH TOÁN ĐỘ CHÍNH XÁC & VẼ MA TRẬN NHẦM LẪN
acc = accuracy_score(query_labels_true, query_labels_pred)
print(f"\nĐỘ CHÍNH XÁC k-NN (Accuracy @ K={K}): {acc * 100:.2f}%\n")

# Lấy danh sách các món ăn duy nhất
classes = sorted(list(set(labels)))

cm = confusion_matrix(query_labels_true, query_labels_pred, labels=classes)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)

plt.title(f'Ma trận nhầm lẫn trên tập Kiểm thử (k-NN với K={K})', fontsize=16, fontweight='bold')
plt.xlabel('AI Dự đoán (Predicted)', fontsize=12)
plt.ylabel('Thực tế (True)', fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base_dir, "figures")
plot_path = os.path.join(out_dir, "confusion_matrix.png")
plt.savefig(plot_path, dpi=300)
print(f"Đã lưu ma trận nhầm lẫn tại: {plot_path}")