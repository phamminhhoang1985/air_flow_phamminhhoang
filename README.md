# Báo Cáo Bài Tập MLOps — End-to-End Pipeline: Airflow + MLflow + FastAPI Serving

**Môn học:** DDM501 — AI in DevOps, DataOps, MLOps  
**Mô tả bài tập:** Xây dựng hệ thống MLOps hoàn chỉnh từ khâu trích xuất, kiểm định dữ liệu tự động (Data Pipeline với Airflow), huấn luyện & đăng ký phiên bản mô hình (MLflow Model Registry) đến phục vụ dự đoán trực tuyến (FastAPI Serving API).

---

## 🏛 1. Kiến Trúc Hệ Thống (Architecture)

Hệ thống được thiết kế dạng Microservices đóng gói hoàn toàn trong Docker với 3 dịch vụ chính:

```
[ Nguồn Dữ Liệu Thô ] (wdbc.csv)
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 🔄 Airflow Data & ML Pipeline (wdbc_pipeline)                          │
│                                                                        │
│ 1. ingest  ➔ 2. validate ➔ 3. split ➔ 4. scale ➔ 5. train_and_register ➔ 6. report │
└────────────────────────────────────────────────┬───────────────────────┘
                                                 │ (Gửi Model & Metrics)
                                                 ▼
                               ┌───────────────────────────────────┐
                               │ 🧪 MLflow Tracking & Registry     │
                               │    Port: 15010 (SQLite + Artifact)│
                               └─────────────────┬─────────────────┘
                                                 │ (Load Model qua URI)
                                                 ▼
                               ┌───────────────────────────────────┐
                               │ 🚀 FastAPI Serving API            │
                               │    Port: 18011 (/health, /predict)│
                               └───────────────────────────────────┘
```

### Chi tiết các dịch vụ trong `docker-compose.yml`:
| Dịch vụ | Port Public | Nhiệm vụ chính |
|---|---|---|
| **`airflow`** | `18080:8080` | Quản lý và lập lịch chạy DAG `wdbc_pipeline` (Ingest, Validate, Split, Scale, Train, Report). |
| **`mlflow`** | `15010:5000` | Quản lý các thí nghiệm (Experiment Tracking), Artifact và Model Registry (`breast-cancer-classifier`). |
| **`api`** | `18011:8000` | Microservice phục vụ dự đoán REST API, tự động nạp mô hình từ MLflow Registry qua URI `models:/breast-cancer-classifier/<VERSION>`. |

---

## 🔄 2. Luồng Xử Lý Của Airflow DAG (`wdbc_pipeline`)

DAG `wdbc_pipeline` bao gồm **6 task thực thi tự động**:
1. **`ingest`**: Trích xuất dữ liệu thô và lưu bản chụp (**Snapshot**) tại `data/staging/<ds>/raw.parquet`.
2. **`validate`**: Kiểm tra chất lượng dữ liệu (null, giá trị âm, trùng lặp, outlier). Đạt cơ chế **Fail-Fast** với `AirflowFailException` nếu tỷ lệ lỗi vượt quá 5% (bỏ qua retry tự động để tiết kiệm tài nguyên).
3. **`split`**: Phân chia dữ liệu Train (80%) / Test (20%) dựa trên **SHA-256 Hash của `sample_id`** (đảm bảo tính nhất quán `deterministic` 100%).
4. **`scale`**: Chuẩn hóa Z-score. Tính tham số `mean/std` trên duy nhất tập **Train** để tránh rò rỉ dữ liệu (**Data Leakage**).
5. **`train_and_register`**: Huấn luyện mô hình `RandomForestClassifier`, tính các chỉ số (`ROC AUC`, `Accuracy`, `F1-Score`), gửi log thông số lên MLflow Tracking và đăng ký phiên bản mới vào **MLflow Model Registry**.
6. **`report`**: Ghi nhận nhật ký lượt chạy vào `summary.json` và nối thêm dòng báo cáo vào `history.jsonl`.

---

## 🚀 3. Hướng Dẫn Khởi Chạy & Kiểm Thử Chi Tiết

### **Bước 1: Khởi tạo tất cả dịch vụ bằng Docker**
Chạy lệnh sau tại thư mục dự án:
```bash
docker compose up -d --build
```
Kiểm tra trạng thái các container:
```bash
docker compose ps
```
> Chờ khoảng 30–60 giây cho đến khi dịch vụ `mlflow` và `airflow` báo trạng thái **healthy**.

Lấy mật khẩu tài khoản `admin` của Airflow Web UI:
```bash
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt
```

---

### **Bước 2: Kích hoạt Pipeline tự động Train & Đăng ký Mô hình**
Thực thi toàn bộ DAG cho một ngày logic (ví dụ `2026-08-25`):
```bash
docker compose exec airflow airflow dags test wdbc_pipeline 2026-08-25
```
* Kết quả: Tất cả 6 task sẽ chạy hoàn tất. Mô hình `breast-cancer-classifier` **version 1** sẽ được tự động tạo và đăng ký trên MLflow Model Registry.
* Truy cập giao diện MLflow tại: **`http://127.0.0.1:15010`** để xem Experiment & Model Version 1.

---

### **Bước 3: Khởi chạy và Kiểm thử Serving API (Version 1)**
Khởi động/khởi động lại container API để nạp mô hình Version 1 từ Registry:
```bash
docker compose up -d api
```

Kiểm tra trạng thái sức khỏe và thông tin mô hình đang nạp (`/health`):
```bash
curl -s http://localhost:18011/health
```
**Kết quả trả về:**
```json
{"status":"ok","model_loaded":true,"model_uri":"models:/breast-cancer-classifier/1"}
```

Tạo dữ liệu thử nghiệm và gửi yêu cầu dự đoán (`/predict`):
```bash
# Tạo payload JSON mẫu từ script
docker compose exec airflow python scripts/sample_request.py > sample_request.json

# Gửi request POST tới API
curl -s -X POST http://localhost:18011/predict \
  -H "content-type: application/json" -d @sample_request.json
```
**Kết quả dự đoán trả về:**
```json
{
  "prediction": "malignant",
  "probability_benign": 0.0,
  "served_by": "models:/breast-cancer-classifier/1"
}
```

---

### **Bước 4: Kiểm tra cơ chế Fail-Fast của Data Validation**
Cố tình làm hỏng 12% số dòng dữ liệu bằng script:
```bash
docker compose exec airflow python scripts/corrupt_extract.py
```
Kích hoạt lại pipeline:
```bash
docker compose exec airflow airflow dags test wdbc_pipeline 2026-08-26
```
* **Kết quả:** Task `validate` lập tức dừng lượt chạy với thông báo lỗi `12.0% of rows rejected, limit is 5%` và phát sinh `AirflowFailException` để bỏ qua 3 lượt retry lãng phí.

Khôi phục lại dữ liệu gốc:
```bash
docker compose exec airflow python scripts/corrupt_extract.py --repair
```

---

### **Bước 5: Thử nghiệm Nâng cấp / Rollback Phiên bản Mô hình (Version Switch)**
1. Chạy lại DAG cho ngày tiếp theo để đăng ký **Version 2**:
   ```bash
   docker compose exec airflow airflow dags test wdbc_pipeline 2026-08-27
   ```
2. Cập nhật biến môi trường `MODEL_VERSION=2` trong file `.env`:
   ```env
   MODEL_NAME=breast-cancer-classifier
   MODEL_VERSION=2
   ```
3. Cập nhật dịch vụ API:
   ```bash
   docker compose up -d api
   curl -s http://localhost:18011/health
   ```
   **Kết quả:** API chuyển sang phục vụ `"model_uri": "models:/breast-cancer-classifier/2"` mà không cần sửa code hay rebuild Docker Image.

---

## 📤 4. Hướng Dẫn Push Bài Tập Lên GitHub Nộp Cho Giáo Viên

Thực hiện các lệnh Git sau tại thư mục dự án để đẩy toàn bộ mã nguồn lên repository:

```bash
# 1. Kiểm tra các file đã thay đổi
git status

# 2. Thêm tất cả thay đổi vào Staging
git add .

# 3. Commit bài tập với message rõ ràng
git commit -m "Complete MLOps Assignment: Airflow Pipeline + MLflow Registry + FastAPI Serving"

# 4. Push code lên GitHub repository của bạn
git push origin main
```

---
*Hoàn thành bài tập môn DDM501 — AI in DevOps, DataOps, MLOps.*