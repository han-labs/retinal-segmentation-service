import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse 
from fastapi.staticfiles import StaticFiles                 
import io
from contextlib import asynccontextmanager
from core.logic import load_model, run_inference

#Tối ưu hóa: Tải model một lần duy nhất khi server khởi động
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    print("Server shutdown.")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(lifespan=lifespan)

# Phục vụ file tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static")

# Trang chủ
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# API Endpoint để dự đoán 
@app.post("/predict/", response_class=StreamingResponse)
async def predict_image(file: UploadFile = File(...)):
    """Nhận file ảnh, gọi Lõi AI để xử lý, và trả về ảnh kết quả"""
    # Đọc nội dung file ảnh người dùng tải lên
    image_bytes = await file.read()
    
    # Gọi hàm xử lý trung tâm
    result_image_np = run_inference(image_bytes=image_bytes)
    
    # Chuyển đổi ảnh kết quả (NumPy array) thành file PNG
    is_success, img_encoded = cv2.imencode(".png", result_image_np)
    if not is_success:
        return {"error": "Could not encode image"}, 500

    # Trả về file PNG cho người dùng
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")