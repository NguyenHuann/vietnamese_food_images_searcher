import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import layers
from tqdm import tqdm

# 1. ĐỊNH NGHĨA LỚP GeM POOLING

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


# 2. CẤU HÌNH ĐƯỜNG DẪN TỪ TERMINAL (ARGPARSE)
parser = argparse.ArgumentParser(description="Trích xuất vector đặc trưng từ tập ảnh đồ ăn.")
parser.add_argument("--input", type=str, default="../dataset", help="Đường dẫn tới thư mục dataset")
parser.add_argument("--output", type=str, default="../models", help="Thư mục lưu kết quả .pkl")
args = parser.parse_args()

DATASET_DIR = args.input
OUTPUT_DIR = args.output
MODEL_FILE = os.path.join(OUTPUT_DIR, "vietnamese_food_feature_extractor.keras")

# Tự động tạo thư mục đầu ra nếu chưa có
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 3. TẢI MÔ HÌNH
print("1. Đang tải mô hình AI với lớp GeMPooling...")
model = load_model(
    MODEL_FILE,
    compile=False,
    custom_objects={'GeMPooling': GeMPooling}
)

vectors = []
paths = []

# 4. QUÁ TRÌNH TRÍCH XUẤT ĐẶC TRƯNG
print("2. Bắt đầu trích xuất Vector cho toàn bộ cơ sở dữ liệu ảnh...")
# Duyệt qua tất cả các thư mục con và file
for root, _, files in os.walk(DATASET_DIR):
    # Lọc ra các file ảnh hợp lệ
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Nếu thư mục có ảnh, hiển thị thanh tiến trình (progress bar)
    if image_files:
        for file in tqdm(image_files, desc=f"Đang xử lý thư mục {os.path.basename(root)}"):
            full_path = os.path.join(root, file)
            # Tạo đường dẫn tương đối (Ví dụ: pho/img1.jpg) để dùng cho Web sau này
            relative_path = os.path.relpath(full_path, DATASET_DIR).replace("\\", "/")

            try:
                # 4.1 Load ảnh và ép về đúng kích thước đã train (224x224)
                img = keras_image.load_img(full_path, target_size=(224, 224))
                x = keras_image.img_to_array(img)
                x = np.expand_dims(x, axis=0)  # Thêm chiều batch (1, 224, 224, 3)

                # 4.2 Trích xuất vector
                # Lưu ý: model.predict mặc định sẽ chạy ở inference mode (training=False),
                # nên nó tự động bỏ qua Dropout và các lớp làm méo ảnh (RandomZoom, RandomFlip...)
                vector = model.predict(x, verbose=0)[0]

                vectors.append(vector)
                paths.append(relative_path)
            except Exception as e:
                print(f"Bỏ qua file lỗi {full_path}: {e}")

# 5. LƯU KẾT QUẢ XUỐNG Ổ CỨNG
print("3. Đang đóng gói dữ liệu...")
vectors_save_path = os.path.join(OUTPUT_DIR, "vectors.pkl")
paths_save_path = os.path.join(OUTPUT_DIR, "paths.pkl")

with open(vectors_save_path, "wb") as f:
    pickle.dump(np.array(vectors), f)

with open(paths_save_path, "wb") as f:
    pickle.dump(paths, f)

print(f"\n[THÀNH CÔNG] Đã chuyển đổi {len(vectors)} bức ảnh.")
print(f"Dữ liệu đã được lưu tại thư mục: {OUTPUT_DIR}")