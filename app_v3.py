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
    img_bytes = file.read()  # Đọc file ảnh dưới dạng byte

    def generate():
        try:
            # --- BƯỚC 1: TIỀN XỬ LÝ ---
            yield json.dumps({"step": "1. Đang nạp và làm sạch ảnh..."}) + "\n"
            time.sleep(0.5)  # Giả lập chờ để người dùng kịp đọc

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_resized = img.resize((224, 224))  # Đổi thành 224 nếu dùng v5
            x = keras_image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)

            # --- BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG ---
            yield json.dumps({"step": "2. AI đang quét đặc trưng món ăn..."}) + "\n"
            time.sleep(0.5)
            query_vector = model.predict(x, verbose=0)[0]

            # --- BƯỚC 3: TÌM KIẾM CSDL ---
            yield json.dumps({"step": "3. Đang so sánh với 10.000 ảnh trong CSDL..."}) + "\n"
            time.sleep(0.5)
            similarities = np.dot(db_vectors, query_vector)

            top_5_indices = np.argsort(similarities)[::-1][:5]
            top_5_sims = similarities[top_5_indices]

            # Temperature Scaling
            TEMPERATURE = 0.5
            exp_sims = np.exp(top_5_sims / TEMPERATURE)
            confidence_scores = exp_sims / np.sum(exp_sims) * 100
            mon_nuoc_group = ["pho_bo", "bun_bo_hue", "bun_thang", "bun_rieu", "hu_tieu"]

            results = []
            for i, idx in enumerate(top_5_indices):
                path = db_paths[idx]
                raw_folder = path.split("/")[0]
                dish_name = DISH_VN_NAMES.get(raw_folder, raw_folder.replace("_", " ").title())

                warning_msg = ""
                if i == 0 and raw_folder in mon_nuoc_group:
                    diff = confidence_scores[0] - confidence_scores[1]
                    if diff < 15.0:
                        warning_msg = "⚠️ Ảnh hơi mờ/khó đoán, có thể nhầm với món nước khác!"

                results.append({
                    "dish_name": dish_name,
                    "similarity": float(confidence_scores[i]),
                    "warning": warning_msg,
                    "image_url": f"/dataset/{path}"
                })

            # --- BƯỚC 4: HOÀN TẤT VÀ TRẢ KẾT QUẢ ---
            yield json.dumps({"step": "Hoàn tất!", "results": results}) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    # Trả về Response dưới dạng NDJSON (Newline Delimited JSON)
    return Response(generate(), mimetype='application/x-ndjson')


if __name__ == "__main__":
    print("Server đang chạy tại: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)