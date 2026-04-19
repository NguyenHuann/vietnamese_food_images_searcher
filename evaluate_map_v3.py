import pickle
import numpy as np
from tqdm import tqdm

K = 5  # Số lượng ảnh trả về muốn đánh giá (Top-50)

print("Đang tải cơ sở dữ liệu Vector...")
with open("vectors_v3.pkl", "rb") as f:
    vectors = pickle.load(f)  # Mảng numpy shape (N, 512)
with open("paths_v3.pkl", "rb") as f:
    paths = pickle.load(f)

# Trích xuất nhãn gốc từ đường dẫn (ví dụ: 'bun_bo_hue/img1.jpg' -> 'bun_bo_hue')
true_labels = [path.split("/")[0] for path in paths]
N = len(vectors)

print(f"Bắt đầu đánh giá mAP@{K} trên tổng số {N} bức ảnh...")

map_at_k = 0.0
total_hits = 0  # Đếm tổng số lượng ảnh trả về đúng

for i in tqdm(range(N)):
    query_vec = vectors[i]
    query_label = true_labels[i]

    # Tính Cosine Similarity của ảnh hiện tại với TOÀN BỘ database
    sims = np.dot(vectors, query_vec)

    # Sắp xếp từ cao xuống thấp
    sorted_indices = np.argsort(sims)[::-1]

    # LỌC: Bỏ qua chính bức ảnh đang dùng làm Query (vì giống 100%)
    retrieved_indices = [idx for idx in sorted_indices if idx != i][:K]

    # Tính AP@K cho bức ảnh này
    hits = 0
    sum_prec = 0.0

    for rank, idx in enumerate(retrieved_indices):
        retrieved_label = true_labels[idx]

        # Nếu nhãn của ảnh trả về TRÙNG với nhãn của ảnh gốc
        if retrieved_label == query_label:
            hits += 1
            total_hits += 1
            # Precision tại vị trí rank hiện tại
            sum_prec += hits / (rank + 1.0)

    # Giả định K <= Tổng số ảnh của món đó trong DB
    ap_k = sum_prec / K
    map_at_k += ap_k

# Tính Mean Average Precision (Trung bình của tất cả AP)
map_at_k = (map_at_k / N) * 100
# Tính độ chính xác thuần túy (Precision@K trung bình)
precision_at_k = (total_hits / (N * K)) * 100

print("\n" + "=" * 40)
print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN {N} ẢNH:")
print(f"- Precision@{K} trung bình : {precision_at_k:.2f}% (Trong 5 ảnh trả về, trung bình có bao nhiêu ảnh đúng)")
print(f"- mAP@{K} tổng thể        : {map_at_k:.2f}% (Độ đo tối thượng, ưu tiên ảnh đúng nằm ở Top 1-2)")
print("=" * 40)