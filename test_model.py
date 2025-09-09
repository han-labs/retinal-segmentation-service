# test_model.py
import cv2
from core.logic import run_inference

IMAGE_PATH = "sample_image.jpg"

def run_test():
    print("--- KIỂM THỬ MÔ HÌNH---")
    
    # Đọc file ảnh dưới dạng bytes, giả lập việc người dùng upload
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    # Gọi hàm xử lý trung tâm
    result_image = run_inference(image_bytes=image_bytes)
    
    # Lưu kết quả
    output_path = "result_image_refactored.png"
    cv2.imwrite(output_path, result_image)
    print(f"[*] Đã lưu ảnh kết quả vào file: {output_path}")
    print("--- KIỂM THỬ HOÀN TẤT ---")

if __name__ == "__main__":
    run_test()