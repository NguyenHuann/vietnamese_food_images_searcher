import os
import io
import json
import time
from flask import Response
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
from collections import Counter

# Cấu hình Flask
app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# 1. Tải Model và Database Vector lên RAM
print("Đang khởi động Server và nạp AI...")
model = load_model("vietnamese_food_feature_extractor_v3.keras", compile=False)

with open("vectors_v3.pkl", "rb") as f:
    db_vectors = pickle.load(f)
with open("paths_v3.pkl", "rb") as f:
    db_paths = pickle.load(f)

# Từ điển ánh xạ tên món
DISH_VN_NAMES = {
    "banh_cuon": "Bánh Cuốn", "bun_bo_hue": "Bún Bò Huế", "pho": "Phở",
    "bun_rieu": "Bún Riêu", "banh_bot_loc": "Bánh Bột Lọc", "banh_can": "Bánh Căn",
    "banh_canh": "Bánh Canh", "banh_khot": "Bánh Khọt", "banh_mi": "Bánh Mì",
    "banh_trang_nuong": "Bánh Tráng Nướng", "banh_xeo": "Bánh Xèo",
    "bun_dau_mam_tom": "Bún Đậu Mắm Tôm", "bun_mam": "Bún Mắm", "bun_thit_nuong": "Bún Thịt Nướng",
    "canh_cua": "Canh Cua", "chao_long": "Cháo Lòng", "com_tam": "Cơm Tấm",
    "goi_cuon": "Gỏi Cuốn", "hu_tieu": "Hủ Tiếu", "mi_quang": "Mì Quảng",
}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/dataset/<path:filename>")
def serve_image(filename):
    return send_from_directory(DATASET_DIR, filename)


@app.route("/search", methods=["POST"])
def search():
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file"}), 400

    k_param = request.form.get("k", 5)
    try:
        K = int(k_param)
    except ValueError:
        K = 5

    file = request.files["file"]
    img_bytes = file.read()

    def generate():
        try:
            yield json.dumps({"step": "1. Đang xử lý ảnh..."}) + "\n"
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_resized = img.resize((224, 224))
            x = keras_image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)

            yield json.dumps({"step": "2. AI đang trích xuất đặc trưng..."}) + "\n"
            query_vector = model.predict(x, verbose=0)[0]

            yield json.dumps({"step": "3. Đang tìm và tiến hành bầu chọn (k-NN)..."}) + "\n"
            similarities = np.dot(db_vectors, query_vector)
            top_k_indices = np.argsort(similarities)[::-1][:K]
            top_k_sims = similarities[top_k_indices]

            # --- CƠ CHẾ BẦU CHỌN (MAJORITY VOTING) ---
            # Lấy tối đa 10 ảnh đầu tiên để bầu chọn
            vote_count = min(10, K)
            vote_labels = [db_paths[idx].split("/")[0] for idx in top_k_indices[:vote_count]]

            # Đếm xem thư mục nào xuất hiện nhiều nhất
            counter = Counter(vote_labels)
            majority_folder, majority_votes = counter.most_common(1)[0]
            predicted_dish = DISH_VN_NAMES.get(majority_folder, majority_folder.replace("_", " ").title())

            confidence_scores = np.clip(top_k_sims, 0.0, 1.0) * 100

            # Chỉ giữ lại duy nhất 1 vòng lặp để tạo mảng kết quả
            results = []
            for i, idx in enumerate(top_k_indices):
                path = db_paths[idx]
                raw_folder = path.split("/")[0]
                dish_name = DISH_VN_NAMES.get(raw_folder, raw_folder.replace("_", " ").title())

                results.append({
                    "dish_name": dish_name,
                    "similarity": float(confidence_scores[i]),
                    "is_correct": bool(raw_folder == majority_folder),
                    "image_url": f"/dataset/{path}"
                })

            # Gửi thêm thông tin kết quả bầu chọn về cho Web
            yield json.dumps({
                "step": "Hoàn tất!",
                "predicted_dish": predicted_dish,
                "majority_votes": majority_votes,
                "vote_count": vote_count,
                "results": results
            }) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return Response(generate(), mimetype='application/x-ndjson')


if __name__ == "__main__":
    print("Server đang chạy tại: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)