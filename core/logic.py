import torch
import cv2
import numpy as np
from PIL import Image
import io

from core.models.unet_plus_plus import create_model
from core.datasets.base_dataset import get_valid_transform

# Biến toàn cục để lưu trữ model, tránh tải lại nhiều lần
LOADED_MODEL = None
DEVICE = None

def load_model():
    """
    Tải model và lưu vào biến toàn cục. Chỉ thực hiện một lần.
    """
    global LOADED_MODEL, DEVICE
    if LOADED_MODEL is None:
        print("[*] Lần đầu tiên: Đang tải model vào bộ nhớ...")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = create_model()
        model = model.to(DEVICE)
        
        model_path = "model_weights/stage4_final_model.pth"
        checkpoint = torch.load(model_path, map_location=DEVICE)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        LOADED_MODEL = model
        print(f"=> Model đã được tải thành công lên thiết bị: {DEVICE}")
    return LOADED_MODEL, DEVICE

def run_inference(image_bytes: bytes) -> np.ndarray:
    """
    Hàm xử lý hoàn chỉnh: nhận bytes ảnh -> trả về ảnh kết quả (NumPy array).
    """
    model, device = load_model()

    # 1. Đọc và chuẩn hóa ảnh đầu vào
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)

    # 2. Áp dụng các phép biến đổi (transform) 
    transform = get_valid_transform()
    transformed = transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # 3. Chạy dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        
        # Chuyển kết quả về dạng ảnh NumPy
        pred_np = (output.cpu().numpy() > 0.5).astype(np.uint8)
        pred_np = pred_np[0, 0] # Trích xuất ảnh từ batch

    # 4. Hậu xử lý: Resize về kích thước gốc
    original_h, original_w = image_np.shape[:2]
    result_image = cv2.resize(pred_np, (original_w, original_h))
    result_image_binary = (result_image > 0.5).astype(np.uint8) * 255
    
    return result_image_binary