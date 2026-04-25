import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==========================================
# 1. CẤU HÌNH NÂNG CAO
# ==========================================
BASE_PATH = "/kaggle/input/datasets/anos22/vietnamese-food-dataset"
IMG_SIZE = (224, 224)
# Ép Batch Size lớn để thuật toán Batch Hard Triplet Loss phát huy sức mạnh.
# (Nếu GPU báo lỗi cạn RAM - OOM, hãy giảm xuống 32 nhé)
BATCH_SIZE = 64
VECTOR_DIM = 512


def find_data_path(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'train' in dirs and 'val' in dirs: return root
    return base_path


DATA_DIR = find_data_path(BASE_PATH)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")


# ==========================================
# 2. HÀM TRIPLET LOSS (BATCH HARD MINING)
# ==========================================
def custom_triplet_loss(margin=1.0):
    def triplet_loss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        embeddings = y_pred
        dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)
        square_norm = tf.linalg.diag_part(dot_product)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
        distances = tf.maximum(distances, 0.0)
        labels_equal = tf.equal(tf.expand_dims(y_true, 0), tf.expand_dims(y_true, 1))
        mask_positive = tf.cast(labels_equal, tf.float32) - tf.eye(tf.shape(y_true)[0])
        mask_negative = 1.0 - tf.cast(labels_equal, tf.float32)
        hardest_positive_dist = tf.reduce_max(distances * mask_positive, axis=1, keepdims=True)
        max_dist = tf.reduce_max(distances, axis=1, keepdims=True)
        distances_plus_max = distances + max_dist * (1.0 - mask_negative)
        hardest_negative_dist = tf.reduce_min(distances_plus_max, axis=1, keepdims=True)
        loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        return tf.reduce_mean(loss)

    return triplet_loss


# ==========================================
# 3. PIPELINE DỮ LIỆU
# ==========================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(height_factor=(-0.3, -0.2)),
    layers.RandomContrast(0.2),
])

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# ==========================================
# 4. CUSTOM LAYER: GeM POOLING (Thay cho GAP)
# Đăng ký class để sau này Keras load model không bị lỗi
# ==========================================
@tf.keras.utils.register_keras_serializable()
class GeMPooling(layers.Layer):
    def __init__(self, p=3.0, eps=1e-6, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.p = tf.Variable(p, trainable=True, dtype=tf.float32)
        self.eps = eps

    def call(self, x):
        x = tf.maximum(x, self.eps)
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        x = tf.pow(x, 1.0 / self.p)
        return x

    def get_config(self):
        config = super(GeMPooling, self).get_config()
        config.update({"p": self.p.numpy(), "eps": self.eps})
        return config


# ==========================================
# 5. XÂY DỰNG KIẾN TRÚC MÔ HÌNH
# ==========================================
base_model = EfficientNetB2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)

# Áp dụng GeM Pooling để khuếch đại đặc trưng nổi bật
x = GeMPooling(p=3.0)(x)

x = layers.Dropout(0.5)(x)
x = layers.Dense(VECTOR_DIM, activation=None)(x)
outputs = layers.UnitNormalization(axis=1)(x)

model = tf.keras.Model(inputs, outputs)

# ==========================================
# 6. CHIẾN THUẬT 2 GIAI ĐOẠN (TWO-STAGE FINE-TUNING)
# ==========================================

# --- GIAI ĐOẠN 1: WARM-UP (LÀM NÓNG) ---
print("\n[PHASE 1] BẮT ĐẦU WARM-UP: Đóng băng EfficientNet, chỉ train phần đầu...")
base_model.trainable = False  # Đóng băng 100% xương sống

# Dùng AdamW với Learning Rate khá lớn để khởi tạo trọng số Dense
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=custom_triplet_loss(margin=1.0)
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5  # Chỉ cần 5 epoch để làm nóng
)

# --- GIAI ĐOẠN 2: FINE-TUNING (TINH CHỈNH SÂU) ---
print("\n[PHASE 2] BẮT ĐẦU FINE-TUNING: Mở khóa 50 lớp cuối để học chi tiết món ăn...")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Re-compile với Learning Rate RẤT NHỎ để không phá hỏng kiến thức cũ
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-4),
    loss=custom_triplet_loss(margin=1.0)
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint("/kaggle/working/best_model_advanced.keras", monitor='val_loss', save_best_only=True)
]

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=45,  # Chạy nốt 45 Epochs còn lại
    callbacks=callbacks
)

# Lưu bản cuối cùng
model.save("/kaggle/working/vietnamese_food_extractor_v4.keras")
print("\n--- HOÀN TẤT HUẤN LUYỆN NÂNG CAO ---")