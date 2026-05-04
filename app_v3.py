import os
import io
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from PIL import Image
from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


# ==========================================
# 1. ĐỊNH NGHĨA LỚP GeM POOLING (INLINE ĐỂ SERVER ĐỘC LẬP)
# ==========================================
@tf.keras.utils.register_keras_serializable()
class GeMPooling(layers.Layer):
    def __init__(self, p=3.0, eps=1e-6, trainable_p=True, **kwargs):
        super().__init__(**kwargs)
        self.p_init = p
        self.eps = eps
        self.trainable_p = trainable_p

    def build(self, input_shape):
        self.p = self.add_weight(
            name='gem_power',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.p_init),
            trainable=self.trainable_p,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.maximum(inputs, self.eps)
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        x = tf.pow(x, 1.0 / self.p)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p_init,
            "eps": self.eps,
            "trainable_p": self.trainable_p
        })
        return config


# ==========================================
# 2. CẤU HÌNH FLASK & ĐƯỜNG DẪN
# ==========================================
app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_DIR = r'C:\Users\ADMIN\Downloads\archive\dataset'
# Tên các file AI mới nhất của bạn (Hãy để chung thư mục với app.py)
MODEL_FILE = os.path.join(BASE_DIR, "vietnamese_food_feature_extractor_v4.keras")
VECTOR_FILE = os.path.join(BASE_DIR, "vectors_v4.pkl")
PATH_FILE = os.path.join(BASE_DIR, "paths_v4.pkl")

# ==========================================
# 3. NẠP MÔ HÌNH VÀ CƠ SỞ DỮ LIỆU
# ==========================================
print("Đang khởi động Server và nạp mô hình...")

# Load mô hình với lớp Custom GeMPooling
try:
    model = load_model(MODEL_FILE, compile=False, custom_objects={'GeMPooling': GeMPooling})
    print("Đã nạp thành công mô hình AI.")
except Exception as e:
    print(f"Lỗi nạp mô hình: {e}")

# Load cơ sở dữ liệu Vector
try:
    with open(VECTOR_FILE, "rb") as f:
        db_vectors = pickle.load(f)
    with open(PATH_FILE, "rb") as f:
        db_paths = pickle.load(f)
    print(f"✅ Đã nạp thành công {len(db_vectors)} vector đặc trưng.")
except Exception as e:
    print(f"❌ Lỗi nạp dữ liệu vector: {e}")

# Từ điển ánh xạ tên món
DISH_VN_NAMES = {
    "banh_cuon": "Bánh Cuốn", "bun_bo_hue": "Bún Bò Huế", "pho": "Phở",
    "bun_rieu": "Bún Riêu", "banh_bot_loc": "Bánh Bột Lọc", "banh_can": "Bánh Căn",
    "banh_canh": "Bánh Canh", "banh_khot": "Bánh Khọt", "banh_mi": "Bánh Mì",
    "banh_trang_nuong": "Bánh Tráng Nướng", "banh_xeo": "Bánh Xèo",
    "bun_dau_mam_tom": "Bún Đậu Mắm Tôm", "bun_mam": "Bún Mắm", "bun_thit_nuong": "Bún Thịt Nướng",
    "canh_cua": "Canh Cua", "chao_long": "Cháo Lòng", "com_tam": "Cơm Tấm",
    "goi_cuon": "Gỏi Cuốn", "hu_tieu": "Hủ Tiếu", "mi_quang": "Mì Quảng",
    # AI sẽ tự động xử lý các món còn lại trong bộ dữ liệu 30 món bằng cách replace "_" thành dấu cách
}


# ==========================================
# 4. KHỞI TẠO CÁC API
# ==========================================
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
            # Bước 1: Tiền xử lý ảnh
            yield json.dumps({"step": "1. Đang xử lý ảnh..."}) + "\n"
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_resized = img.resize((224, 224))
            x = keras_image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)

            # Bước 2: Trích xuất đặc trưng
            yield json.dumps({"step": "2. AI đang trích xuất đặc trưng..."}) + "\n"
            query_vector = model.predict(x, verbose=0)[0]

            # Bước 3: Thuật toán k-NN và Tính toán độ tương đồng
            yield json.dumps({"step": "3. Đang tìm và tiến hành bầu chọn (k-NN)..."}) + "\n"
            similarities = np.dot(db_vectors, query_vector)
            top_k_indices = np.argsort(similarities)[::-1][:K]
            top_k_sims = similarities[top_k_indices]

            # --- CƠ CHẾ BẦU CHỌN (MAJORITY VOTING) ---
            vote_count = min(10, K)
            vote_labels = [db_paths[idx].split("/")[0] for idx in top_k_indices[:vote_count]]

            # Đếm số phiếu
            counter = Counter(vote_labels)
            majority_folder, majority_votes = counter.most_common(1)[0]

            # Xử lý tên món: Lấy từ từ điển, nếu không có thì tự động format lại thư mục gốc
            predicted_dish = DISH_VN_NAMES.get(majority_folder, majority_folder.replace("_", " ").title())

            confidence_scores = np.clip(top_k_sims, 0.0, 1.0) * 100

            # Xây dựng mảng kết quả
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

            # Trả kết quả cuối cùng cho Client
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