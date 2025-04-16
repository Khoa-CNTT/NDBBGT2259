from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import cv2
import uvicorn
import os

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc ["http://127.0.0.1:5500"] để bảo mật hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa danh sách nhãn biển báo
labels = [
    "Giới hạn tốc độ 20km/h", "Giới hạn tốc độ 30km/h", "Giới hạn tốc độ 50km/h", 
    "Giới hạn tốc độ 60km/h", "Giới hạn tốc độ 70km/h", "Giới hạn tốc độ 80km/h", 
    "Hết giới hạn tốc độ 80km/h", "Giới hạn tốc độ 100km/h", "Giới hạn tốc độ 120km/h", 
    "Cấm vượt", "Cấm xe tải vượt", "Giao nhau với đường không ưu tiên", "Bắt đầu đường ưu tiên", 
    "Nhường đường", "Dừng lại", "Đường cấm", "Cấm xe tải", "Cấm đi ngược chiều", "Cảnh báo nguy hiểm", 
    "Đường cong nguy hiểm bên trái", "Đường cong nguy hiểm bên phải", "Cảnh báo hai khúc cua", "Đường gập nghềnh",
    "Cảnh báo đường trơn trượt", "Đường hẹp bên phải", "Công trường", "Giao thông tín hiệu", "Người đi bộ", 
    "Trẻ em qua đường", "Chú ý xe đạp", "Đường trơn", "Cảnh báo động vật hoang dã", "Hết giới hạn tốc độ", 
    "Rẽ phải bắt buộc", "Rẽ trái bắt buộc", "Đi thẳng", "Đi thẳng hoặc rẽ phải", "Đi thẳng hoặc rẽ trái", 
    "Đi bên phải", "Đi bên trái", "Vòng xuyến", "Hết cấm vượt", "Hết cấm xe tải vượt"
]

# Tải mô hình đã huấn luyện
MODEL_PATH = "traffic_sign_model_tf3.h5"  # Đảm bảo đúng đường dẫn đến mô hình đã lưu
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình: {MODEL_PATH}. Kiểm tra lại đường dẫn.")

model = tf.keras.models.load_model(MODEL_PATH)

# Kiểm tra API hoạt động
@app.get("/")
def home():
    return {"message": "API Nhận dạng biển báo giao thông đang chạy!"}

# API nhận ảnh và dự đoán biển báo
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc ảnh từ file upload
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Kiểm tra ảnh có đọc được không
        if image is None:
            return JSONResponse({"error": "Không thể đọc ảnh. Kiểm tra lại định dạng file."}, status_code=400)

        # Resize ảnh về kích thước mong muốn (64x64 cho mô hình mới)
        image = cv2.resize(image, (64, 64))

        # Chuyển BGR sang RGB (vì OpenCV đọc ảnh ở dạng BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Chuẩn hóa ảnh về [0,1]
        image = image / 255.0  

        # Thêm batch dimension (1, 64, 64, 3)
        image = np.expand_dims(image, axis=0)

        # Dự đoán
        predictions = model.predict(image)

        # Lấy nhãn có xác suất cao nhất
        label_index = np.argmax(predictions)
        confidence = float(predictions[0][label_index])
        label = labels[label_index]

        return JSONResponse({"label": label, "confidence": confidence})
    except Exception as e:
        import traceback
        print("❌ Lỗi khi xử lý ảnh:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# Chạy ứng dụng
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    #python -m uvicorn app:app --reload
