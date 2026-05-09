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
        self.p = self.add_weight(name='gem_power', shape=(), initializer=tf.keras.initializers.Constant(self.p_init),
                                 trainable=self.trainable_p, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        x = tf.maximum(inputs, self.eps)
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        return tf.pow(x, 1.0 / self.p)

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p_init, "eps": self.eps, "trainable_p": self.trainable_p})
        return config

# 2. CLASS QUẢN LÝ VECTOR ĐẶC TRƯNG
class VectorDatabaseManager:
    def __init__(self, vector_file, path_file, model_file=None):
        self.vector_file = vector_file
        self.path_file = path_file
        self.vectors = []
        self.paths = []
        self.model = None
        self.model_file = model_file

        self.load_database()

    def load_database(self):
        """Tải dữ liệu từ file .pkl lên RAM"""
        if os.path.exists(self.vector_file) and os.path.exists(self.path_file):
            with open(self.vector_file, "rb") as f:
                self.vectors = list(pickle.load(f))  # Ép kiểu list để dễ thêm/xóa
            with open(self.path_file, "rb") as f:
                self.paths = pickle.load(f)
            print(f"Đã tải {len(self.vectors)} vector vào bộ nhớ.")
        else:
            print("Database trống hoặc không tìm thấy file. Khởi tạo mới.")

    def save_database(self):
        """Lưu trạng thái hiện tại xuống file .pkl"""
        with open(self.vector_file, "wb") as f:
            pickle.dump(np.array(self.vectors), f)
        with open(self.path_file, "wb") as f:
            pickle.dump(self.paths, f)
        print(f"Đã lưu {len(self.vectors)} vector xuống ổ cứng.")

    def _init_model(self):
        """Chỉ load model AI khi cần trích xuất ảnh mới để tiết kiệm RAM"""
        if self.model is None:
            if not self.model_file or not os.path.exists(self.model_file):
                raise ValueError("Không tìm thấy file model .keras!")
            print("Đang nạp AI model...")
            self.model = load_model(self.model_file, compile=False, custom_objects={'GeMPooling': GeMPooling})

    def add_image(self, image_path, relative_save_path):
        """Thêm 1 ảnh mới vào database mà không cần quét lại từ đầu"""
        if relative_save_path in self.paths:
            print(f"Ảnh {relative_save_path} đã tồn tại trong Database!")
            return False

        self._init_model()
        try:
            img = keras_image.load_img(image_path, target_size=(224, 224))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            vector = self.model.predict(x, verbose=0)[0]

            self.vectors.append(vector)
            self.paths.append(relative_save_path)
            print(f"Đã thêm thành công: {relative_save_path}")
            return True
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh: {e}")
            return False

    def delete_image(self, relative_path):
        """Xóa 1 ảnh khỏi database"""
        try:
            idx = self.paths.index(relative_path)
            self.paths.pop(idx)
            self.vectors.pop(idx)
            print(f"Đã xóa thành công: {relative_path}")
            return True
        except ValueError:
            print(f"Không tìm thấy {relative_path} trong Database.")
            return False

    def info(self):
        """Thống kê chi tiết Database"""
        print("\nTHỐNG KÊ DATABASE:")
        print(f"Tổng số ảnh: {len(self.paths)}")

        # Đếm số lượng ảnh theo từng món ăn
        labels = [p.split('/')[0] for p in self.paths]
        from collections import Counter
        stats = Counter(labels)
        for dish, count in stats.most_common():
            print(f"  - {dish}: {count} ảnh")
        print("=" * 30 + "\n")

    def sync_dataset(self, dataset_dir):
        print(f"\nĐang quét thư mục: {dataset_dir} ...")

        # BƯỚC 1: Tối ưu hóa thuật toán tìm kiếm bằng Set (O(1) lookup)
        # Chuyển danh sách paths hiện tại thành Set để kiểm tra trùng lặp siêu nhanh
        existing_paths = set(self.paths)

        images_to_add = []

        # BƯỚC 2: Thu thập toàn bộ file ảnh có trong ổ cứng
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    full_path = os.path.join(root, file)
                    # Tạo đường dẫn tương đối (VD: pho/img1.jpg)
                    relative_path = os.path.relpath(full_path, dataset_dir).replace("\\", "/")

                    # BƯỚC 3: Đối chiếu - Nếu chưa có mặt trong Database thì đưa vào danh sách chờ
                    if relative_path not in existing_paths:
                        images_to_add.append((full_path, relative_path))

        # BƯỚC 4: Xử lý danh sách chờ
        if not images_to_add:
            print("Database đã được đồng bộ 100%. Không tìm thấy ảnh mới nào!")
            return

        print(f"Phát hiện {len(images_to_add)} ảnh mới. Đang khởi động AI để trích xuất...")
        self._init_model()  # Đảm bảo AI đã được nạp lên RAM

        success_count = 0

        # Dùng thanh tiến trình tqdm để theo dõi
        for full_path, relative_path in tqdm(images_to_add, desc="Tiến trình trích xuất", unit="ảnh"):
            try:
                # Tiền xử lý ảnh
                img = keras_image.load_img(full_path, target_size=(224, 224))
                x = keras_image.img_to_array(img)
                x = np.expand_dims(x, axis=0)

                # Trích xuất vector 512 chiều
                vector = self.model.predict(x, verbose=0)[0]

                # Cập nhật vào RAM
                self.vectors.append(vector)
                self.paths.append(relative_path)
                success_count += 1

            except Exception as e:
                # Dùng tqdm.write thay cho print để không làm vỡ giao diện thanh tiến trình
                tqdm.write(f"Lỗi file {relative_path}: {str(e)}")

        # BƯỚC 5: Tự động lưu Database mới xuống ổ cứng
        if success_count > 0:
            print(f"\nĐã trích xuất và thêm thành công {success_count} ảnh mới!")
            self.save_database()

# 3. KỊCH BẢN SỬ DỤNG MẪU (TESTING)
if __name__ == "__main__":

    db = VectorDatabaseManager(
        vector_file="../models/vectors.pkl",
        path_file="../models/paths.pkl",
        model_file="../models/vietnamese_food_feature_extractor.keras"
    )
    DATASET_DIR = "../dataset"
    db.sync_dataset(DATASET_DIR)
    # Xem thống kê
    db.info()
    # thêm ảnh bằng hàm db.add_image
    # xóa ảnh bằng hàm db.delete_image
