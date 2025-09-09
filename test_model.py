# test_model.py
import cv2
import numpy as np
# Import thêm hàm preprocess_image    
from core.logic import load_trained_model, patch_based_inference, preprocess_image

# --- CẤU HÌNH ---
MODEL_PATH = "model_weights/stage4_final_model.pth"
IMAGE_PATH = "sample_image.jpg"
# --- KẾT THÚC CẤU HÌNH ---

def run_test():
    print("--- BẮT ĐẦU KIỂM THỬ LÕI AI (PHIÊN BẢN CHUẨN CHỈNH) ---")
    
    # 1. Tải model
    model, device = load_trained_model(MODEL_PATH)
    
    # 2. Tải ảnh và chuẩn bị
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"LỖI: Không thể tải ảnh tại đường dẫn: {IMAGE_PATH}")
        return
    
    # Ghi lại kích thước ảnh gốc
    original_h, original_w = image.shape[:2]
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # --- BƯỚC TIỀN XỬ LÝ MỚI ---
    print(f"[*] Kích thước ảnh gốc: {original_w}x{original_h}")
    print("[*] Tiền xử lý: Đệm ảnh để phù hợp với model...")
    image_padded = preprocess_image(image_rgb)
    print(f"[*] Kích thước ảnh sau khi đệm: {image_padded.shape[1]}x{image_padded.shape[0]}")
    # --- KẾT THÚC BƯỚC MỚI ---
    
    # 3. Chạy dự đoán trên ảnh đã đệm
    print("[*] Bắt đầu chạy dự đoán (có thể mất một lúc)...")
    prediction_map_padded = patch_based_inference(model, image_padded, device=device)
    print("=> Chạy dự đoán thành công!")
    
    # 4. Lưu ảnh kết quả
    # --- BƯỚC HẬU XỬ LÝ MỚI ---
    # Cắt lại ảnh kết quả về kích thước gốc
    prediction_map = prediction_map_padded[:original_h, :original_w]
    # --- KẾT THÚC BƯỚC MỚI ---

    threshold = 0.5
    result_image_binary = (prediction_map > threshold).astype(np.uint8) * 255
    output_path = "result_image.png"
    cv2.imwrite(output_path, result_image_binary)
    print(f"[*] Đã lưu ảnh kết quả (kích thước gốc) vào file: {output_path}")
    
    print("--- KIỂM THỬ HOÀN TẤT ---")

if __name__ == "__main__":
    run_test()