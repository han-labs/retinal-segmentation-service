from fastapi import FastAPI

# 1. Khởi tạo ứng dụng FastAPI
# Đây sẽ là "trái tim" của API của bạn
app=FastAPI()

# 2. Định nghĩa một "đường dẫn" (route)
# Dòng này nghĩa là: "Khi có ai đó truy cập vào trang chủ ('/') bằng phương thức GET..."

@app.get("/")
def read_root():
    # "...thì hãy chạy hàm này và trả về một đối tượng JSON"
    return {"message": "Hello, World!"}  