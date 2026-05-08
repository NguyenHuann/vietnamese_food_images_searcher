# Sử dụng Python 3.10 siêu nhẹ
FROM python:3.10-slim

# Chuyển vào thư mục làm việc
WORKDIR /app

# Copy thư viện và cài đặt
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code và model lên
COPY . .

# Mở cổng 7860 theo chuẩn Hugging Face
EXPOSE 7860

# Khởi chạy bằng Gunicorn (như đã bàn ở phần trước)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]