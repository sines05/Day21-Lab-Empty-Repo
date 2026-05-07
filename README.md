# Day 21 Lab: Automated MLOps Pipeline on GCP

Dự án này triển khai một quy trình MLOps hoàn chỉnh cho bài toán dự đoán chất lượng rượu (Wine Quality) sử dụng tập dữ liệu Wine Quality từ UCI. Quy trình bao gồm từ thực nghiệm cục bộ, phiên bản hóa dữ liệu, đến xây dựng pipeline CI/CD tự động và triển khai API Inference trên Google Cloud Platform.

## 🚀 Các Tính Năng Chính

- **CI/CD Pipeline (GitHub Actions)**: Tự động chạy Unit Test, Huấn luyện mô hình, Đánh giá chất lượng và Triển khai.
- **Data Version Control (DVC)**: Quản lý phiên bản dữ liệu lớn và lưu trữ an toàn trên Google Cloud Storage (GCS).
- **Experiment Tracking (MLflow/DagsHub)**: Theo dõi mọi thí nghiệm, chỉ số (Accuracy, F1) và lưu trữ artifacts (Mô hình, Báo cáo).
- **Automated Deployment**: Tự động triển khai mô hình mới nhất lên GCP Compute Engine qua SSH khi vượt qua ngưỡng chất lượng.
- **FastAPI Inference**: Cung cấp API dự đoán thời gian thực với các endpoint `/health` và `/predict`.

## 🏆 Thành Tựu & Bonus Points

- **[Bonus 1] Cloud Storage Integration**: Cấu hình thành công DVC với GCS remote sử dụng Service Account.
- **[Bonus 2 & 3] Advanced Training & Reporting**:
  *   Hỗ trợ đa thuật toán (Random Forest, Gradient Boosting, Logistic Regression).
  *   Tự động tạo báo cáo `report.txt` với Confusion Matrix, Precision, Recall và đẩy lên MLflow Artifacts.
- **[Bonus 4] Quality Gate & Rollback**:
  *   Thiết lập ngưỡng Accuracy >= 0.70 cho Eval Gate.
  *   Cơ chế tự động so sánh mô hình mới và cũ trước khi triển khai.
- **[Bonus 5] Data Drift Monitoring**:
  *   Kiểm tra phân phối nhãn dữ liệu huấn luyện để phát hiện lệch dữ liệu ngay trong pipeline.

## 📊 Kết Quả Cuối Cùng

Mô hình tốt nhất hiện tại sử dụng thuật toán **Gradient Boosting** với các thông số:
- `n_estimators`: 200
- `max_depth`: 5
- **Accuracy đạt được: 0.7160** (Vượt ngưỡng yêu cầu 0.70).

## 🛠 Hướng Dẫn Sử Dụng

### 1. Cài đặt môi trường
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Chạy huấn luyện cục bộ
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python3 src/train.py
```

### 3. Kiểm tra API (Inference)
```bash
curl -X POST http://<VM_IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```
