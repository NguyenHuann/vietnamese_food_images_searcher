import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, average_precision_score
from collections import Counter

# 1. TẢI VÀ CHIA DỮ LIỆU
with open("../models/vectors.pkl", "rb") as f:
    vectors = pickle.load(f)
with open("../models/paths.pkl", "rb") as f:
    paths = pickle.load(f)

labels = [os.path.basename(os.path.dirname(p)) for p in paths]

db_vectors, query_vectors, db_labels, query_labels_true = train_test_split(
    vectors, labels, test_size=0.2, random_state=42, stratify=labels
)

classes = sorted(list(set(labels)))

# 2. CHẠY MÔ PHỎNG k-NN VÀ TÍNH mAP
K = 5
query_labels_pred = []
ap_scores_per_class = {c: [] for c in classes}

print("Đang đánh giá hệ thống, vui lòng đợi...\n")

for q_vec, q_label in zip(query_vectors, query_labels_true):
    # Tính khoảng cách (Cosine/Dot Product)
    sims = np.dot(db_vectors, q_vec)

    # ---- Phần 1: Bầu chọn k-NN để lấy nhãn dự đoán (Dùng cho Acc, Pre, Rec, F1) ----
    top_k_indices = np.argsort(sims)[::-1][:K]
    top_k_labels = [db_labels[idx] for idx in top_k_indices]
    majority_label = Counter(top_k_labels).most_common(1)[0][0]
    query_labels_pred.append(majority_label)

    # ---- Phần 2: Tính Average Precision (Dùng cho mAP) ----
    y_true_binary = np.array([1 if db_label == q_label else 0 for db_label in db_labels])
    if np.sum(y_true_binary) > 0:
        ap = average_precision_score(y_true_binary, sims)
        ap_scores_per_class[q_label].append(ap)

# Tính tổng mAP
map_per_class = {c: np.mean(ap_scores_per_class[c]) if ap_scores_per_class[c] else 0.0 for c in classes}
overall_map = np.mean(list(map_per_class.values()))

# 3. IN BÁO CÁO TỔNG HỢP (CLASSIFICATION REPORT)
print("=" * 60)
print("BẢNG BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH (EVALUATION REPORT)")
print("=" * 60)

# In mAP
print(f"1. mAP (Mean Average Precision) Hệ thống : {overall_map * 100:.2f}%\n")

# In Classification Report (Chứa Acc, Pre, Rec, F1)
print("2. Chi tiết theo từng lớp (Dựa trên dự đoán k-NN):")
report = classification_report(query_labels_true, query_labels_pred, target_names=classes, digits=4)
print(report)

print("=" * 60)