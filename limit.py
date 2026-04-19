import shutil
from pathlib import Path


def copy_limited_files(source_root, target_root, limit=500):
    """
    Sao chép tối đa 'limit' file từ mỗi thư mục con ở source_root
    sang thư mục cùng tên ở target_root.
    """
    source_path = Path(source_root)
    target_path = Path(target_root)

    # Kiểm tra thư mục nguồn có tồn tại không
    if not source_path.exists():
        print(f"Lỗi: Thư mục nguồn '{source_root}' không tồn tại.")
        return

    # Duyệt qua các thư mục con trong thư mục nguồn
    for subfolder in source_path.iterdir():
        if subfolder.is_dir():
            # Tạo đường dẫn tương ứng ở thư mục đích
            destination_folder = target_path / subfolder.name

            # Lấy danh sách tất cả các file trong thư mục con (không bao gồm thư mục)
            files = [f for f in subfolder.iterdir() if f.is_file()]

            # Chỉ lấy tối đa số lượng file theo giới hạn
            files_to_copy = files[:limit]

            if files_to_copy:
                # Tạo thư mục đích nếu chưa có
                destination_folder.mkdir(parents=True, exist_ok=True)

                print(f"Đang sao chép {len(files_to_copy)} file từ '{subfolder.name}'...")

                for file in files_to_copy:
                    try:
                        # Sử dụng copy2 để giữ nguyên metadata (thời gian tạo, v.v.)
                        shutil.copy2(file, destination_folder / file.name)
                    except Exception as e:
                        print(f"Lỗi khi sao chép {file.name}: {e}")
            else:
                print(f"Thư mục '{subfolder.name}' không có file nào để sao chép.")

    print("\n--- Hoàn thành quá trình xử lý ---")


# --- CẤU HÌNH ĐƯỜNG DẪN TẠI ĐÂY ---
SOURCE_DIRECTORY = 'D:\project\searchIMG\dataset'
TARGET_DIRECTORY = 'D:\project\_dataset\dataset'

if __name__ == "__main__":
    copy_limited_files(SOURCE_DIRECTORY, TARGET_DIRECTORY, limit=500)