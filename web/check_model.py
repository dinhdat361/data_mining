# check_model.py

from joblib import load
import numpy as np

# Tên các Feature (phải đúng thứ tự đã train)
FEATURE_COLS = ['G1', 'G2', 'studytime', 'absences', 'failures']
MODEL_FILE = 'linear_model.joblib'

try:
    # 1. Tải mô hình đã lưu từ tệp .joblib
    model = load(MODEL_FILE)
    print(f"Đã tải mô hình thành công từ tệp: {MODEL_FILE}\n")

    # 2. Trích xuất và in Hệ số chặn (Intercept) - Beta 0
    intercept = model.intercept_
    print(f"--- THÔNG SỐ CỦA MÔ HÌNH LINEAR REGRESSION ---")
    print(f"Hệ số chặn (Intercept, β₀): {intercept:.4f}\n")

    # 3. Trích xuất và in các Hệ số (Coefficients) - Beta 1 đến Beta 5
    coefficients = model.coef_
    
    print(f"| {'Feature':<12} | {'Hệ số (β)':<10} | {'Ý nghĩa':<25} |")
    print(f"|{'-'*12}-|{'-'*10}-|{'-'*25}-|")
    
    for feature, coef in zip(FEATURE_COLS, coefficients):
        sign = "Dương (+)" if coef > 0 else "Âm (-)"
        meaning = f"Tăng {coef:.2f} điểm G3/đơn vị" if coef > 0 else f"Giảm {abs(coef):.2f} điểm G3/đơn vị"
        
        print(f"| {feature:<12} | {coef:>10.4f} | {sign} ({meaning:<23}) |")
    
    # 4. In ra Công thức Hồi quy
    print("\n--- CÔNG THỨC HỒI QUY ---")
    formula = f"G3_Dự đoán = {intercept:.4f}"
    for feature, coef in zip(FEATURE_COLS, coefficients):
        formula += f" + ({coef:.4f} * {feature})"
    print(formula)

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy tệp {MODEL_FILE}. Đảm bảo tệp đã được tải về và nằm trong thư mục hiện tại.")