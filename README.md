# RetinaSeg: AI-Powered Retinal Vessel Segmentation

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-blueviolet?logo=opencv&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?logo=bootstrap&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-as-a-Service application that performs high-precision segmentation of blood vessels in retinal fundus images. This project transforms a complex scientific research model into a real-time, interactive web service.

**Live Demo:** `[LINK-TO-YOUR-DEPLOYED-APP-HERE]` 

---

### Key Features

* **Real-time Segmentation:** Upload a retinal image and get the segmented vessel map instantly.
* **High-Accuracy AI Model:** Utilizes a state-of-the-art **U-Net++ architecture** with an **EfficientNet-B4 backbone**.
* **High-Resolution Processing:** Employs a patch-based inference strategy to accurately segment large, detailed medical images without downscaling.
* **Interactive Web Interface:** Modern, responsive UI built with Bootstrap 5, featuring drag-and-drop upload and sample images for easy testing.
* **RESTful API Backend:** A robust backend built with **FastAPI**, serving the AI model efficiently.
* **Dockerized for Portability:** Fully containerized with Docker, ensuring consistent performance across any environment and readiness for cloud deployment.

### Technology Stack

The project is built with a modern, robust technology stack suitable for deploying AI models as a web service.

| Category | Technology |
| :--- | :--- |
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **AI / Data Processing** | PyTorch, OpenCV, Albumentations, Numpy |
| **Frontend** | HTML5, CSS3, JavaScript (ES6), Bootstrap 5 |
| **DevOps & Infrastructure** | Docker, Ubuntu, Git / GitHub |

### Segmentation in Action

Here is an example of the model's high-fidelity segmentation capability.

![Before and After Segmentation](/static/images/image.png)

---

### Getting Started (Local Setup)

To run this project on your local machine, follow these steps:

**Prerequisites:**
* Python 3.10+
* pip & venv

**1. Clone the repository:**
```bash
git clone https://github.com/han-labs/retinal-segmentation-service.git
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
    * LinkedIn: [linkedin-huynh-gia-han]`www.linkedin.com/in/huynh-gia-han`

### License

This project is licensed under the MIT License - see the `LICENSE` file for details.
