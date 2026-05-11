# ==========================================
# BÀI TẬP PYTHON CƠ BẢN - TH5
# ==========================================

# 1. Hàm tính giai thừa (Yêu cầu 25)
def tinh_giai_thua(n):
    if n == 0 or n == 1:
        return 1
    ket_qua = 1
    for i in range(2, n + 1):
        ket_qua *= i
    return ket_qua

# 2. Hàm tính giá trị trung bình của dãy số (Yêu cầu 26)
def tinh_trung_binh(danh_sach):
    if not danh_sach:
        return 0
    return sum(danh_sach) / len(danh_sach)

# 3. Hàm tính lợi nhuận sau 12 tháng (Yêu cầu 27)
# Công thức: Tiền cuối kỳ = Tiền gốc * (1 + Lãi suất)
def tinh_loi_nhuan(tien_goc, lai_suat_thang):
    # Giả sử lãi suất nhập vào dạng số thập phân (ví dụ 0.05 cho 5%)
    # Tính lợi nhuận sau 12 tháng với lãi suất đơn hoặc kép tùy yêu cầu
    # Ở đây dùng lãi kép (compound interest) cho thực tế:
    tien_cuoi_ky = tien_goc * (1 + lai_suat_thang)**12
    loi_nhuan = tien_cuoi_ky - tien_goc
    return tien_cuoi_ky, loi_nhuan

# --- CHƯƠNG TRÌNH CHÍNH (Demo kết quả) ---
if __name__ == "__main__":
    print("--- Kết quả thực hiện bài tập cơ bản ---")
    
    # Test Giai thừa
    n = 5
    print(f"1. Giai thừa của {n} là: {tinh_giai_thua(n)}")
    
    # Test Trung bình dãy số
    day_so = [10, 20, 30, 40, 50]
    print(f"2. Giá trị trung bình của dãy {day_so} là: {tinh_trung_binh(day_so)}")
    
    # Test Lợi nhuận
    von_dau_tu = 100000000  # 100 triệu
    lai_suat = 0.005        # 0.5% mỗi tháng
    tong_tien, lai = tinh_loi_nhuan(von_dau_tu, lai_suat)
    print(f"3. Sau 12 tháng:")
    print(f"   - Tổng số tiền nhận được: {tong_tien:,.0f} VND")
    print(f"   - Tiền lãi thuần: {lai:,.0f} VND")