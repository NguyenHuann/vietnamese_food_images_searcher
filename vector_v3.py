import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tqdm import tqdm

# Cấu hình
MODEL_PATH = "vietnamese_food_feature_extractor_v3.keras"
DATASET_DIR = "./dataset"  # Thư mục chứa ảnh gốc của bạn

print("1. Đang tải mô hình...")
model = load_model(MODEL_PATH, compile=False)

vectors = []
paths = []

print("2. Bắt đầu trích xuất Vector cho toàn bộ ảnh...")
for root, _, files in os.walk(DATASET_DIR):
    for file in tqdm(files, desc=f"Đang xử lý thư mục {os.path.basename(root)}"):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            # Lưu đường dẫn tương đối (VD: bun_bo_hue/img1.jpg) để Web dễ đọc
            relative_path = os.path.relpath(full_path, DATASET_DIR).replace("\\", "/")

            try:
                # Load và xử lý ảnh giống lúc train (kích thước 160x160)
                img = keras_image.load_img(full_path, target_size=(224, 224))
                x = keras_image.img_to_array(img)
                x = np.expand_dims(x, axis=0)

                # Trích xuất vector
                vector = model.predict(x, verbose=0)[0]

                vectors.append(vector)
                paths.append(relative_path)
            except Exception as e:
                print(f"Lỗi ảnh {full_path}: {e}")

# Lưu thành file
with open("vectors_v3.pkl", "wb") as f:
    pickle.dump(np.array(vectors), f)
with open("paths_v3.pkl", "wb") as f:
    pickle.dump(paths, f)

print(f"\n[THÀNH CÔNG] Đã lưu {len(vectors)} vector vào vectors.pkl và paths.pkl")