import math
import pandas as pd
import matplotlib.pyplot as plt

# --- YÊU CẦU 3: CÁC BÀI TOÁN CƠ BẢN ---
print("--- KẾT QUẢ CÁC BÀI TOÁN CƠ BẢN ---")
# 1. Tính giai thừa
print(f"1. Giai thừa của 6 là: {math.factorial(6)}")

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
    # --- PHẦN VẼ BIỂU ĐỒ ĐỂ PHÂN TÍCH ---
    labels = ['Binh thuong', 'No xau (Gian lan)']
    sizes = [tong_kh - no_xau[1], no_xau[1]]
    colors = ['#66b3ff', '#ff9999'] # Xanh cho khách tốt, Đỏ cho nợ xấu
    explode = (0, 0.1)  # Đẩy phần nợ xấu ra một chút để gây chú ý

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
    autopct='%1.2f%%', shadow=True, startangle=140)
    plt.title('Phan tich Ty le Rui ro Tin dung')
    plt.axis('equal') 

    # Tự động lưu biểu đồ thành file ảnh để chèn vào báo cáo Word
    plt.savefig('bieu_do_phan_tich.png')
    print("✅ Da tao bieu do phan tich 'bieu_do_phan_tich.png'!")

    # Hiển thị biểu đồ lên màn hình
    plt.show()
    
    print("\n[Nhận xét]: Dữ liệu cho thấy tỉ lệ rủi ro tín dụng cần được giám sát chặt chẽ.")
except Exception as e:
    print(f"❌ Lỗi: {e}")