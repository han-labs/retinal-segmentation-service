# RetinaSeg: AI-Powered Retinal Vessel Segmentation

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-as-a-Service application that performs high-precision segmentation of blood vessels in retinal fundus images. This project transforms a complex scientific research model into a real-time, interactive web service.

**Live Demo:** `[LINK-TO-YOUR-DEPLOYED-APP-HERE]` *(Bạn sẽ điền link sau khi chúng ta deploy)*

---

### Key Features

* **Real-time Segmentation:** Upload a retinal image and get the segmented vessel map instantly.
* **High-Accuracy AI Model:** Utilizes a state-of-the-art **U-Net++ architecture** with an **EfficientNet-B4 backbone**.
* **High-Resolution Processing:** Employs a patch-based inference strategy to accurately segment large, detailed medical images without downscaling.
* **Interactive Web Interface:** Modern, responsive UI built with Bootstrap 5, featuring drag-and-drop upload and sample images for easy testing.
* **RESTful API Backend:** A robust backend built with **FastAPI**, serving the AI model efficiently.
* **Dockerized for Portability:** Fully containerized with Docker, ensuring consistent performance across any environment and readiness for cloud deployment.

### Technology Stack

| Frontend | Backend | AI/ML | Deployment |
| :--- |:--- |:--- |:--- |
| HTML5 | Python 3.11 | PyTorch | Docker |
| CSS3 | FastAPI | OpenCV | Ubuntu Server |
| JavaScript (ES6) | Uvicorn | Albumentations | Git / GitHub |
| Bootstrap 5 | | | |

### Segmentation in Action

Here is an example of the model's high-fidelity segmentation capability.

![Before and After Segmentation](/static/images/image.png)
*(Lưu ý: Đường dẫn này hoạt động khi xem trên web, nhưng để GitHub hiển thị, bạn cần đảm bảo ảnh đã được đẩy lên repo)*

---

### Getting Started (Local Setup)

To run this project on your local machine, follow these steps:

**Prerequisites:**
* Python 3.10+
* pip & venv

**1. Clone the repository:**
```bash
git clone [https://github.com/han-labs/retinal-segmentation-service.git](https://github.com/han-labs/retinal-segmentation-service.git)
cd retinal-segmentation-service
```

**2. Create and activate a virtual environment:**
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the application:**
```bash
uvicorn main:app --reload
```
The application will be running at `http://127.0.0.1:8000`.

### Author

* **Huynh Gia Han**
    * GitHub: [@han-labs](https://github.com/han-labs)
    * LinkedIn: `[LINK-TO-YOUR-LINKEDIN-PROFILE]`

### License

This project is licensed under the MIT License - see the `LICENSE` file for details.
