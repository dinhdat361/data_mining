# app.py
from flask import Flask, request, render_template
from joblib import load
import pandas as pd
import numpy as np # Sử dụng cho việc làm tròn và giới hạn

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# --- Tải Mô hình Đã Huấn Luyện ---
MODEL_FILE = 'linear_model.joblib'
try:
    # Tải mô hình (chỉ 1 lần khi server khởi động)
    model = load(MODEL_FILE)
    print("Mô hình Linear Regression đã được tải thành công!")
except FileNotFoundError:
    print(f"LỖI: KHÔNG tìm thấy tệp {MODEL_FILE}. Vui lòng kiểm tra lại.")
    model = None

# Tên các Feature (Đảm bảo đúng thứ tự đã train)
FEATURE_COLS = ['G1', 'G2', 'studytime', 'absences', 'failures']

# Hàm đánh giá và phân loại (Dựa trên Business Rules)
def analyze_prediction(predicted_g3, failures, absences, studytime):
    # 1. Phân loại theo Điểm Dự đoán (Thang 20)
    if predicted_g3 > 14:
        score_group = "✅ Thành tích Tốt (Dự kiến G3 > 14)"
    elif predicted_g3 >= 10:
        score_group = "🟡 Trung bình/Ổn định (Dự kiến G3 từ 10 - 14)"
    else:
        score_group = "🚨 Rủi ro Cao (Dự kiến G3 < 10)"

    # 2. Phân tích Yếu tố Hành vi (Risk Factors)
    risk_factors = []
    
    # Rủi ro 1: Lịch sử thất bại
    if failures >= 1:
        risk_factors.append(f"⚠️ Rủi ro Lịch sử: Từng rớt {int(failures)} môn trước.")
    
    # Rủi ro 2: Thiếu kỷ luật (mức vắng cao hơn trung bình ~5.7)
    if absences > 5:
        risk_factors.append(f"⚠️ Rủi ro Kỷ luật: Số buổi vắng cao ({int(absences)} buổi).")
    
    # Rủi ro 3: Hiệu suất học (studytime thấp hoặc cao quá mức)
    if studytime <= 1:
        risk_factors.append("⚠️ Rủi ro Nỗ lực: Thời gian học quá thấp (≤ 2h/tuần).")
    elif studytime >= 4 and predicted_g3 < 12:
        # Phát hiện studytime cao nhưng điểm thấp (vấn đề hiệu suất)
        risk_factors.append("🟡 Phân tích Hiệu suất: Nỗ lực cao (≥ 10h/tuần) nhưng điểm chưa tương xứng (cần cải thiện phương pháp).")
        
    if not risk_factors:
        risk_factors.append("👍 Sinh viên ổn định, không có yếu tố rủi ro hành vi đáng kể.")

    return score_group, risk_factors

# --- Định tuyến (Routing) ---

# Trang chủ - Hiển thị form nhập liệu
@app.route('/')
def home():
    # Render trang HTML, cung cấp giá trị mặc định cho form
    default_values = {'g1': 12, 'g2': 13, 'studytime': 2, 'absences': 4, 'failures': 0}
    return render_template('index.html', **default_values)

# API dự đoán - Xử lý POST request từ form
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Lỗi: Mô hình chưa được tải.", 500
        
    try:
        # Lấy dữ liệu từ form (tất cả đều là string, cần chuyển sang float)
        data = [
            float(request.form['g1']),
            float(request.form['g2']),
            float(request.form['studytime']),
            float(request.form['absences']),
            float(request.form['failures'])
        ]

        # Tạo DataFrame để đảm bảo thứ tự và cấu trúc inputs đúng với mô hình đã train
        input_df = pd.DataFrame([data], columns=FEATURE_COLS)
        
        # Thực hiện dự đoán
        prediction = model.predict(input_df)[0]
        
        # Làm tròn điểm dự đoán và giới hạn trong khoảng [0, 20]
        final_g3 = max(0, min(20, round(prediction)))
        
        # Phân tích kết quả
        score_group, risk_factors = analyze_prediction(
            final_g3, 
            data[4], # failures
            data[3], # absences
            data[2]  # studytime
        )
        
        # Trả kết quả về trang HTML, giữ lại giá trị đã nhập
        return render_template('index.html', 
                                prediction_text=f'{final_g3} / 20',
                                score_group=score_group,
                                risk_factors=risk_factors,
                                g1=data[0], g2=data[1], studytime=data[2], absences=data[3], failures=data[4])

    except ValueError:
        # Xử lý lỗi nếu người dùng nhập ký tự không phải số
        return render_template('index.html', error_message='Dữ liệu nhập vào không hợp lệ. Vui lòng kiểm tra các trường.')
    except Exception as e:
        # Xử lý lỗi hệ thống
        return render_template('index.html', error_message=f'Lỗi hệ thống không xác định: {str(e)}')

if __name__ == '__main__':
    # Chạy ứng dụng web
    app.run(debug=True)