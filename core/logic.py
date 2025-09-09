# core/logic.py

import torch
import cv2
import numpy as np
import albumentations as A
  
# Import các hàm và class cần thiết từ code gốc
# Đường dẫn đã được điều chỉnh cho phù hợp với cấu trúc project của chúng ta
from core.models.unet_plus_plus import create_model# ===================================================================

#  HÀM TIỀN XỬ LÝ ẢNH     

def preprocess_image(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Đệm ảnh để có kích thước chia hết cho 32."""
    transform = A.Compose([
        A.PadIfNeeded(
            min_height=target_size, 
            min_width=target_size, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0
        )
    ])
    processed_image = transform(image=image)['image']
    return processed_image

# HÀM TẢI MODEL
# (Trích xuất và đơn giản hóa từ predict.py và evaluate_hrf_metrics.py)
# ===================================================================

def load_trained_model(model_path):
    """Tải model đã huấn luyện từ file .pth."""
    print(f"[*] Đang tải model từ: {model_path}")
    
    # Xác định thiết bị (chạy trên CPU nếu không có GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Sử dụng thiết bị: {device}")
    
    # Tạo kiến trúc model
    model = create_model()
    
    # Tải trọng số đã huấn luyện
    checkpoint = torch.load(model_path, map_location=device)
    
    # Xử lý các định dạng checkpoint khác nhau
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval() # Chuyển model sang chế độ dự đoán (rất quan trọng!)
    
    print("=> Tải model thành công!")
    return model, device

# ===================================================================
# HÀM DỰ ĐOÁN
# (Trích xuất từ batch_predict.py - đây là hàm xử lý cốt lõi)
# ===================================================================
def patch_based_inference(model, image, patch_size=512, overlap=128, device='cpu'):
    """
    Chạy dự đoán trên ảnh lớn bằng cách chia thành các patch nhỏ.
    Input:
        - model: model PyTorch đã được tải.
        - image: ảnh đầu vào dưới dạng mảng NumPy (H, W, C).
    Output:
        - prediction: bản đồ segment dưới dạng mảng NumPy (H, W).
    """
    model.eval()
    h, w = image.shape[:2]
    
    # Chuẩn hóa ảnh và chuyển thành tensor
    image_normalized = image.astype(np.float32) / 255.0
    
    # Tạo một "canvas" để chứa kết quả
    prediction = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    step = patch_size - overlap
    
    with torch.no_grad():
        # Lặp qua các patch
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Xác định tọa độ của patch để không bị tràn ra ngoài ảnh
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                y_start = y_end - patch_size
                x_start = x_end - patch_size
                
                # Trích xuất patch
                patch = image_normalized[y_start:y_end, x_start:x_end]
                
                # Chuyển patch thành tensor
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
                patch_tensor = patch_tensor.to(device)
                
                # Chạy dự đoán
                output = model(patch_tensor)
                patch_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
                
                # Cộng kết quả vào canvas
                prediction[y_start:y_end, x_start:x_end] += patch_pred
                count_map[y_start:y_end, x_start:x_end] += 1
    
    # Lấy trung bình ở những vùng patch chồng lên nhau
    count_map[count_map == 0] = 1
    prediction = prediction / count_map
    
    return prediction