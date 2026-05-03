import math
import pandas as pd

# --- YÊU CẦU 3: CÁC BÀI TOÁN CƠ BẢN ---
print("--- KẾT QUẢ CÁC BÀI TOÁN CƠ BẢN ---")
# 1. Tính giai thừa
print(f"1. Giai thừa của 5 là: {math.factorial(5)}")

# 2. Tính trung bình cộng
lst = [10, 20, 30, 40, 50]
print(f"2. Trung bình cộng của dãy {lst} là: {sum(lst)/len(lst)}")

# 3. Tính lãi kép (10 triệu, lãi 1%/tháng sau 12 tháng)
von = 10000000
print(f"3. Tiền lãi kép nhận được sau 12 tháng: {von * (1 + 0.01)**12 - von:,.0f} VND")
print("-" * 40)

# --- YÊU CẦU 2: PHÂN TÍCH DỮ LIỆU ---
print("--- KẾT QUẢ PHÂN TÍCH DỮ LIỆU ---")
try:
    # Đọc file excel, header nằm ở dòng thứ 2 (index 1)
    df = pd.read_excel('data.xls', header=1)
    print("✅ Đã đọc dữ liệu thành công!")
    
    # Thống kê cơ bản
    tong_kh = len(df)
    no_xau = df['default payment next month'].value_counts()
    
    print(f"Tổng số khách hàng phân tích: {tong_kh}")
    print(f"Số lượng khách hàng nợ xấu (gian lận): {no_xau[1]}")
    print(f"Tỉ lệ nợ xấu: {(no_xau[1]/tong_kh)*100:.2f}%")
    
    print("\n[Nhận xét]: Dữ liệu cho thấy tỉ lệ rủi ro tín dụng cần được giám sát chặt chẽ.")
except Exception as e:
    print(f"❌ Lỗi: {e}")