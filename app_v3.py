import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# Cấu hình Flask
app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# 1. Tải Model và Database Vector lên RAM
print("Đang khởi động Server và nạp AI...")
model = load_model("vietnamese_food_feature_extractor_v3.keras", compile=False)

with open("vectors_v3.pkl", "rb") as f:
    db_vectors = pickle.load(f)  # Mảng numpy shape (N, 512)
with open("paths_v3.pkl", "rb") as f:
    db_paths = pickle.load(f)

# Từ điển ánh xạ tên món
DISH_VN_NAMES = {
    "banh_cuon": "Bánh Cuốn", "bun_bo_hue": "Bún Bò Huế", "pho": "Phở",
    "bun_rieu": "Bún Riêu",
    "banh_bot_loc": "Bánh Bột Lọc", "banh_can": "Bánh Căn", "banh_canh":"Bánh Canh",
    "banh_khot": "Bánh Khọt", "banh_mi":"Bánh Mì", "banh_trang_nuong":"Bánh Tráng Nướng",
    "banh_xeo":"Bánh Xèo", "bun_dau_mam_tom":"Bún Đậu Mắm Tôm", "bun_mam":"Bún Mắm", "bun_thit_nuong":"Bún Thịt Nướng",
    "canh_cua":"Canh Cua", "chao_long":"Cháo Lòng", "com_tam": "Cơm Tấm", "goi_cuon":"Gỏi Cuốn", "hu_tieu":"Hủ Tiếu",
    "mi_quang":"Mì Quảng",
}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/dataset/<path:filename>")
def serve_image(filename):
    # Trả về ảnh gốc để Frontend hiển thị
    return send_from_directory(DATASET_DIR, filename)


@app.route("/search", methods=["POST"])
def search():
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file"}), 400

    file = request.files["file"]
    try:
        # Đọc ảnh người dùng tải lên
        img = Image.open(file).convert("RGB")
        # đầu vào cũ 160,160
        img_resized = img.resize((224, 224))
        x = keras_image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)

        # Trích xuất vector cho ảnh tìm kiếm
        query_vector = model.predict(x, verbose=0)[0]

        # Thuật toán Tra cứu: Tính Cosine Similarity (Dot Product vì đã L2 Normalize)
        similarities = np.dot(db_vectors, query_vector)

        # Lấy Top 5 ảnh có độ tương đồng cao nhất
        top_5_indices = np.argsort(similarities)[::-1][:50]

        results = []
        for idx in top_5_indices:
            path = db_paths[idx]
            raw_folder = path.split("/")[0]  # Lấy tên thư mục
            dish_name = DISH_VN_NAMES.get(raw_folder, raw_folder.replace("_", " ").title())

            results.append({
                "dish_name": dish_name,
                "similarity": float(similarities[idx]),
                "image_url": f"/dataset/{path}"
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Server đang chạy tại: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)