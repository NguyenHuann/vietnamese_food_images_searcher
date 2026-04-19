import os
import shutil
import random
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ TỈ LỆ CHIA
# ==========================================
INPUT_DIR = "dataset"  # Thư mục gốc chứa các thư mục món ăn hiện tại
OUTPUT_DIR = "dataset_train"  # Thư mục mới sau khi chia

# Tỉ lệ chia (Train, Val, Test). Tổng phải bằng 1.0
# Khuyến nghị: 70% Train, 20% Val, 10% Test
SPLIT_RATIO = (0.7, 0.2, 0.1)

# Seed để đảm bảo mỗi lần chia đều ra kết quả ngẫu nhiên giống nhau (dễ tái lập)
RANDOM_SEED = 42


# ==========================================
# 2. HÀM CHIA DỮ LIỆU
# ==========================================
def split_data(input_dir, output_dir, split_ratio, seed):
    random.seed(seed)

    # Tạo các thư mục gốc (train, val, test)
    sets = ['train', 'val', 'test']
    for s in sets:
        os.makedirs(os.path.join(output_dir, s), exist_ok=True)

    # Lấy danh sách các thư mục con (các nhãn món ăn)
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    print(f"Đang xử lý {len(classes)} nhóm món ăn...\n")

    for class_name in classes:
        class_path = os.path.join(input_dir, class_name)

        # Lấy tất cả các file ảnh hợp lệ trong thư mục của món ăn
        valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = [f for f in os.listdir(class_path) if os.path.splitext(f)[1].lower() in valid_exts]

        # Xáo trộn ngẫu nhiên danh sách ảnh
        random.shuffle(images)

        # Tính toán số lượng ảnh cho từng tập
        total_images = len(images)
        train_count = int(total_images * split_ratio[0])
        val_count = int(total_images * split_ratio[1])

        # Cắt mảng (Slicing) để chia ảnh
        train_imgs = images[:train_count]
        val_imgs = images[train_count: train_count + val_count]
        test_imgs = images[train_count + val_count:]

        # Tạo thư mục cho món ăn này trong cả 3 tập train, val, test
        for s in sets:
            os.makedirs(os.path.join(output_dir, s, class_name), exist_ok=True)

        # Hàm copy file kèm thanh tiến trình
        def copy_files(file_list, set_name):
            dest_dir = os.path.join(output_dir, set_name, class_name)
            for img in tqdm(file_list, desc=f"{class_name} -> {set_name}", leave=False):
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_dir, img)
                shutil.copy(src, dst)

        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'val')
        copy_files(test_imgs, 'test')

        print(f"✔️ {class_name}: {len(train_imgs)} Train | {len(val_imgs)} Val | {len(test_imgs)} Test")

    print(f"\n[THÀNH CÔNG] Đã chia xong dữ liệu. Vui lòng kiểm tra thư mục '{output_dir}'.")


# ==========================================
# 3. CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    assert sum(SPLIT_RATIO) == 1.0, "Tổng các tỉ lệ chia phải bằng 1.0"
    split_data(INPUT_DIR, OUTPUT_DIR, SPLIT_RATIO, RANDOM_SEED)