import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cấu hình file
FILE_NAME = 'Financial Statement Anomaly Dataset.csv'

def run_fraud_report():
    print("="*60)
    print("HỆ THỐNG PHÂN TÍCH VÀ ĐÁNH GIÁ RỦI RO TÀI CHÍNH")
    print("="*60)

    if not os.path.exists(FILE_NAME):
        print(f"Lỗi: Không tìm thấy file {FILE_NAME}!")
        return

    # 1. Đọc và xử lý dữ liệu
    df = pd.read_csv(FILE_NAME)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    data_clean = df[numeric_cols].fillna(0)
    
    # 2. Chạy mô hình phát hiện gian lận (Isolation Forest)
    # Ngưỡng giả định 5% dữ liệu có dấu hiệu bất thường
    model = IsolationForest(contamination=0.05, random_state=42)
    df['label'] = model.fit_predict(data_clean) # 1: Bình thường, -1: Nghi vấn
    
    # 3. Trích xuất số liệu báo cáo
    total_records = len(df)
    anomalies = df[df['label'] == -1]
    num_anomalies = len(anomalies)
    fraud_percentage = (num_anomalies / total_records) * 100

    # XUẤT SỐ LIỆU RA TERMINAL
    print(f"\n[+] KẾT QUẢ PHÂN TÍCH SỐ LIỆU:")
    print(f"  - Tổng số bản ghi kiểm tra: {total_records} dòng")
    print(f"  - Số lượng hồ sơ nghi vấn gian lận: {num_anomalies} dòng")
    print(f"  - Tỷ lệ phần trăm nghi vấn: {fraud_percentage:.2f}%")
    print("-" * 60)
    
    # DÒNG KẾT LUẬN TỔNG QUAN (Theo yêu cầu của bạn)
    status = "CÓ" if num_anomalies > 0 else "KHÔNG"
    print(f"KẾT LUẬN: Hệ thống xác định {status} dấu hiệu gian lận tài chính, "
          f"chiếm tỷ lệ {fraud_percentage:.2f}% trên tổng quy mô dữ liệu.")
    print("="*60)

    # 4. VẼ BIỂU ĐỒ ĐƠN GIẢN (Biểu đồ tròn thể hiện tỷ lệ)
    print("\n[Hệ thống] Đang khởi tạo biểu đồ trực quan...")
    
    plt.figure(figsize=(10, 6))
    labels = ['Bình thường', 'Nghi vấn gian lận']
    sizes = [total_records - num_anomalies, num_anomalies]
    colors = ['#66b3ff', '#ff9999'] # Xanh và Đỏ
    explode = (0, 0.1)  # Làm nổi bật phần gian lận

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Tỷ lệ phân bổ rủi ro trong báo cáo tài chính")
    plt.axis('equal') 
    
    plt.show()

if __name__ == "__main__":
    run_fraud_report()