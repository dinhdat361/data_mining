# test_imports.py

try:
    import flask
    print("✅ Thư viện Flask đã được cài đặt.")
except ImportError:
    print("❌ Lỗi: Thư viện Flask CHƯA được cài đặt. Vui lòng chạy: pip install flask")

try:
    import pandas
    print("✅ Thư viện Pandas đã được cài đặt.")
except ImportError:
    print("❌ Lỗi: Thư viện Pandas CHƯA được cài đặt. Vui lòng chạy: pip install pandas")
    
try:
    import sklearn
    print("✅ Thư viện Scikit-learn đã được cài đặt.")
except ImportError:
    print("❌ Lỗi: Thư viện Scikit-learn CHƯA được cài đặt. Vui lòng chạy: pip install scikit-learn")

try:
    import joblib
    print("✅ Thư viện Joblib đã được cài đặt.")
except ImportError:
    print("❌ Lỗi: Thư viện Joblib CHƯA được cài đặt. Vui lòng chạy: pip install joblib")
    
print("\n--- KIỂM TRA HOÀN TẤT ---")