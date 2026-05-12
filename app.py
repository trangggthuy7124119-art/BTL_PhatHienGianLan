import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình giao diện Web
st.set_page_config(page_title="AI Fraud Detector", layout="wide")

st.title("🛡️ Hệ thống AI Dự báo & Phát hiện Gian lận Tài chính")
st.markdown("---")

# 1. Tải dữ liệu
uploaded_file = st.file_uploader("Tải lên file Báo cáo Tài chính (CSV)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Kiểm tra dữ liệu hợp lệ
        if len(numeric_cols) == 0:
            st.error("❌ File không chứa cột số liệu. Vui lòng kiểm tra lại!")
        elif len(df) == 0:
            st.error("❌ File trống. Vui lòng tải file khác!")
        else:
            # Sidebar: Cấu hình tham số
            st.sidebar.header("Cấu hình Thuật toán")
            threshold = st.sidebar.slider("Ngưỡng rủi ro (%)", 1, 20, 5) / 100
            
            # 2. Xử lý chính
            data = df[numeric_cols].fillna(0)
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # --- THUẬT TOÁN 1: ISOLATION FOREST (Phát hiện dị biệt) ---
            model_iso = IsolationForest(contamination=threshold, random_state=42)
            df['anomaly'] = model_iso.fit_predict(data_scaled)
            df['risk_score'] = -model_iso.decision_function(data_scaled)
            
            # --- THUẬT TOÁN 2: LINEAR REGRESSION (Dự báo sai số) ---
            # Giả sử dùng cột đầu tiên dự báo cột thứ hai để tìm điểm bất thường
            if len(numeric_cols) >= 2:
                X_reg = data[[numeric_cols[0]]]
                y_reg = data[numeric_cols[1]]
                reg = LinearRegression().fit(X_reg, y_reg)
                df['prediction_error'] = np.abs(y_reg - reg.predict(X_reg))

            # 3. Hiển thị số liệu tổng quan
            col1, col2, col3 = st.columns(3)
            total = len(df)
            anomalies = len(df[df['anomaly'] == -1])
            risk_pct = (anomalies / total) * 100

            col1.metric("Tổng hồ sơ", total)
            col2.metric("Số ca nghi vấn", anomalies, delta_color="inverse")
            col3.metric("Nguy cơ hệ thống", f"{risk_pct:.2f}%")

            # 3.5. TIÊU CHÍ ĐÁNH GIÁ RỦI RO THỐNG KÊ
            st.subheader("📋 Tiêu chí đánh giá chứng minh rủi ro Gian lận")
            
            # Tính các chỉ số thống kê
            mean_values = data.mean()
            std_values = data.std()
            
            # Phát hiện outliers bằng IQR (Interquartile Range)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers_count = ((data < (Q1 - 1.5*IQR)) | (data > (Q3 + 1.5*IQR))).sum().sum()
            
            # Hệ số biến thiên (Coefficient of Variation)
            cv_values = (std_values / (mean_values.abs() + 1e-10)) * 100
            
            # Tính skewness và kurtosis
            skewness_values = data.skew()
            kurtosis_values = data.kurtosis()
            
            # Hiển thị các tiêu chí
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("📊 Số điểm Outliers", outliers_count)
            col_b.metric("📈 Thay đổi trung bình (%)", f"{cv_values.mean():.2f}%")
            col_c.metric("⚠️ Phương sai cao nhất", f"{std_values.max():.2f}")
            col_d.metric("🔢 Số cột bất thường", len(skewness_values[skewness_values.abs() > 2]))
            
            # Chi tiết các tiêu chí
            with st.expander("📌 Chi tiết Tiêu chí đánh giá"):
                st.write("#### Phân tích theo từng cột:")
                
                eval_data = []
                for col in numeric_cols:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    cv_val = (std_val / (abs(mean_val) + 1e-10)) * 100
                    skew_val = data[col].skew()
                    kurt_val = data[col].kurtosis()
                    
                    # Đếm outliers trong cột này
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    outlier_cnt = ((data[col] < q1 - 1.5*iqr) | (data[col] > q3 + 1.5*iqr)).sum()
                    
                    eval_data.append({
                        "Cột": col,
                        "Giá trị TBình": f"{mean_val:.2f}",
                        "Độ lệch": f"{std_val:.2f}",
                        "Hệ số CV (%)": f"{cv_val:.2f}",
                        "Outliers": outlier_cnt,
                        "Skewness": f"{skew_val:.2f}",
                        "Kurtosis": f"{kurt_val:.2f}"
                    })
                
                eval_df = pd.DataFrame(eval_data)
                st.dataframe(eval_df, use_container_width=True)
                
                st.write("**Giải thích các tiêu chí:**")
                st.markdown("""
                - **Hệ số CV (%)**: Tỷ lệ độ lệch so với giá trị trung bình. Giá trị cao (>30%) chỉ dữ liệu bất ổn
                - **Outliers**: Số lượng giá trị nằm ngoài khoảng IQR (chỉ dấu hiệu bất thường)
                - **Skewness**: Độ xiên của phân bố. |Skewness| > 2 cho thấy phân bố rất lệch
                - **Kurtosis**: Độ nhọn. Kurtosis cao chỉ có nhiều giá trị cực đoan
                """)
            
            # 4. Trực quan hóa
            st.subheader("📊 Biểu đồ phân tích rủi ro")
            tab1, tab2 = st.tabs(["Phân bổ rủi ro", "Tương quan số liệu"])
            
            with tab1:
                fig, ax = plt.subplots()
                sns.histplot(df['risk_score'], kde=True, color="red", ax=ax)
                st.pyplot(fig)
                
            with tab2:
                if len(numeric_cols) >= 2:
                    fig2, ax2 = plt.subplots()
                    sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], hue='anomaly', palette={1:'#0066ff', -1:'#ff3333'}, ax=ax2)
                    st.pyplot(fig2)
                else:
                    st.info("⚠️ Cần ít nhất 2 cột số liệu để vẽ biểu đồ tương quan.")

            # 5. Kết luận
            st.subheader("📝 Kết luận tổng quan")
            if risk_pct > 10:
                st.error(f"CẢNH BÁO: Tỷ lệ gian lận cao ({risk_pct:.2f}%). Cần kiểm tra khẩn cấp!")
            else:
                st.success(f"Hệ thống ổn định. Tỷ lệ rủi ro nằm trong ngưỡng cho phép ({risk_pct:.2f}%).")

            # 6. KHUYẾN NGHỊ & HÀNH ĐỘNG
            st.subheader("💡 Khuyến nghị & Hành động cần thực hiện")
            
            if risk_pct > 15:
                rec_level = "🔴 **NGUY HIỂM CAO**"
                rec_actions = [
                    "🚨 Tiến hành kiểm toán toàn bộ hệ thống ngay lập tức",
                    "📞 Liên hệ với bộ phận kiểm thẩm dữ liệu",
                    "🔒 Phong tỏa các giao dịch đáng nghi ngay",
                    "📊 Đối chiếu với dữ liệu lịch sử để tìm mẫu"
                ]
            elif risk_pct > 10:
                rec_level = "🟠 **RỦI RO CAO**"
                rec_actions = [
                    "⚠️ Tiến hành đánh giá chi tiết hồ sơ nghi vấn",
                    "📋 Tăng cường kiểm tra các cột có CV > 30%",
                    "🔍 Xem xét lại quy trình nhập liệu",
                    "📞 Trao đổi với các bộ phận liên quan"
                ]
            elif risk_pct > 5:
                rec_level = "🟡 **RỦI RO TRUNG BÌNH**"
                rec_actions = [
                    "📌 Giám sát thường xuyên các hồ sơ nghi vấn",
                    "📊 Rà soát định kỳ hằng tháng",
                    "📈 Theo dõi xu hướng biến động dữ liệu",
                    "💾 Cập nhật mô hình phát hiện gian lận"
                ]
            else:
                rec_level = "🟢 **RỦI RO THẤP**"
                rec_actions = [
                    "✅ Dữ liệu trong tình trạng tốt",
                    "📅 Tiếp tục giám sát thường xuyên",
                    "📊 Rà soát định kỳ quý",
                    "📚 Cập nhật kiến thức về phát hiện gian lận"
                ]
            
            st.write(f"**Mức độ rủi ro hiện tại: {rec_level}**")
            st.write("**Hành động đề xuất:**")
            for action in rec_actions:
                st.write(f"- {action}")
            
            # 7. HỌC TẬP & CẢI THIỆN
            st.subheader("🎓 Học tập & Cải thiện chất lượng dữ liệu")
            
            with st.expander("📚 Hướng dẫn phát hiện gian lận tài chính"):
                st.markdown("""
                ### 🎯 **7 Dấu hiệu cảnh báo gian lận tài chính:**

                1. **🔴 Dữ liệu bất thường lớn**
                   - Các giá trị vượt ngoài khoảng bình thường (outliers)
                   - Biểu hiện của việc làm giả hoặc thao tác dữ liệu
                   
                2. **📊 Mẫu phân bố lệch (Skewness cao)**
                   - Dữ liệu không tuân theo phân bố chuẩn
                   - Cho thấy có nhóm dữ liệu không tự nhiên
                   
                3. **⚖️ Độ biến thiên cao (CV > 30%)**
                   - Dữ liệu không ổn định, diff bất ngờ tăng/giảm
                   - Báo hiệu kiểm soát chất lượng kém
                   
                4. **📉 Mối quan hệ phi tuyến tính**
                   - Các biến không có mối liên hệ logic
                   - Chỉ ra sự can thiệp nhân tạo
                   
                5. **🔢 Tương quan bất thường**
                   - Các cột không nên liên quan lại có mối liên hệ
                   - Dấu hiệu của hành vi gian lận có kế hoạch
                   
                6. **⏰ Thay đổi đột ngột**
                   - Khoảng thời gian cụ thể có nhiều bất thường
                   - Gợi ý có sự kiện/hành động cụ thể xảy ra
                   
                7. **👤 Mẫu hành vi bất thường**
                   - Cùng người lập nhiều giao dịch nghi vấn
                   - Chỉ ra khả năng liên quan tới gian lận
                
                ---
                
                ### 📋 **Quy trình rà soát dữ liệu tài chính an toàn:**
                
                1. **Định nghĩa chỉ tiêu bình thường** - Xác định khoảng giá trị hợp lệ
                2. **Thu thập dữ liệu lịch sử** - Để so sánh và phát hiện bất thường
                3. **Chạy phân tích thống kê** - Tính CV, Skewness, Kurtosis
                4. **Phát hiện outliers** - Dùng phương pháp IQR hoặc Z-score
                5. **Phân loại rủi ro** - Chia thành LOW/MED/HIGH
                6. **Kiểm tra thủ công** - Xác minh lại các hồ sơ nghi vấn
                7. **Cập nhật mô hình** - Học từ các trường hợp thực tế
                
                """)
            
            with st.expander("🔧 Đề xuất cải thiện hệ thống"):
                st.markdown("""
                ### 🎯 **Kế hoạch cải thiện 3 tháng:**
                
                **TUẦN 1-2:** Kiểm thẩm cơ sở dữ liệu
                - Xác nhận tính chính xác của dữ liệu hiện tại
                - Làm sạch dữ liệu lỗi/trùng lặp
                
                **TUẦN 3-4:** Xây dựng quy trình kiểm soát
                - Thiết lập các ngưỡng cảnh báo
                - Huấn luyện đội ngũ về phát hiện gian lận
                
                **TUẦN 5-8:** Triển khai hệ thống tự động
                - Tích hợp AI để phát hiện bất thường
                - Tạo báo cáo định kỳ
                
                **TUẦN 9-12:** Tối ưu hóa & bảo trì
                - Điều chỉnh mô hình theo kết quả thực tế
                - Cập nhật quy trình dựa trên phản hồi
                
                ### 📊 **Chỉ số KPI cần theo dõi:**
                - ✅ Tỷ lệ phát hiện chính xác (Precision)
                - ✅ Tỷ lệ bao phủ (Recall)
                - ✅ Thời gian phát hiện trung bình
                - ✅ Chi phí xử lý mỗi trường hợp
                """)
            
            st.write("---")
            
            # 8. ĐÁNH GIÁ CHI TIẾT KẾT QUẢ RÀ SOÁT
            st.subheader("📊 Đánh giá chi tiết kết quả rà soát")
            
            anomaly_df = df[df['anomaly'] == -1].sort_values(by='risk_score', ascending=False)
            
            if len(anomaly_df) > 0:
                # Phân loại mức độ rủi ro
                def classify_risk(score):
                    if score >= 0.8:
                        return "🔴 NGUY HIỂM CẤP 1"
                    elif score >= 0.6:
                        return "🟠 NGUY HIỂM CẤP 2"
                    elif score >= 0.4:
                        return "🟡 RỦI RO CAO"
                    else:
                        return "🟢 RỦI RO TRUNG BÌNH"
                
                anomaly_df = anomaly_df.copy()
                anomaly_df['Mức độ rủi ro'] = anomaly_df['risk_score'].apply(classify_risk)
                
                # Thống kê theo mức độ
                st.write("#### 📈 Thống kê hồ sơ nghi vấn theo mức độ rủi ro:")
                
                risk_counts = anomaly_df['Mức độ rủi ro'].value_counts()
                col1, col2, col3, col4 = st.columns(4)
                
                if "🔴 NGUY HIỂM CẤP 1" in risk_counts.index:
                    col1.metric("🔴 Nguy hiểm cấp 1", risk_counts.get("🔴 NGUY HIỂM CẤP 1", 0))
                else:
                    col1.metric("🔴 Nguy hiểm cấp 1", 0)
                    
                if "🟠 NGUY HIỂM CẤP 2" in risk_counts.index:
                    col2.metric("🟠 Nguy hiểm cấp 2", risk_counts.get("🟠 NGUY HIỂM CẤP 2", 0))
                else:
                    col2.metric("🟠 Nguy hiểm cấp 2", 0)
                    
                if "🟡 RỦI RO CAO" in risk_counts.index:
                    col3.metric("🟡 Rủi ro cao", risk_counts.get("🟡 RỦI RO CAO", 0))
                else:
                    col3.metric("🟡 Rủi ro cao", 0)
                    
                if "🟢 RỦI RO TRUNG BÌNH" in risk_counts.index:
                    col4.metric("🟢 Rủi ro TBình", risk_counts.get("🟢 RỦI RO TRUNG BÌNH", 0))
                else:
                    col4.metric("🟢 Rủi ro TBình", 0)
                
                # Hiển thị danh sách chi tiết
                st.write("#### 📋 Danh sách chi tiết hồ sơ nghi vấn:")
                
                # Tạo bảng hiển thị với thêm cột Mức độ rủi ro
                display_cols = list(anomaly_df.columns)
                if 'anomaly' in display_cols:
                    display_cols.remove('anomaly')
                if 'prediction_error' in display_cols:
                    display_cols.remove('prediction_error')
                
                display_df = anomaly_df[['Mức độ rủi ro', 'risk_score'] + [c for c in display_cols if c not in ['Mức độ rủi ro', 'risk_score']]].copy()
                display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # 9. ĐỀ XUẤT HÀNH ĐỘNG THEO MỨC ĐỘ
                st.subheader("💼 Đề xuất hành động cụ thể")
                
                tab_action1, tab_action2, tab_action3, tab_action4 = st.tabs(
                    ["🔴 Cấp 1", "🟠 Cấp 2", "🟡 CAO", "📋 Tóm tắt"]
                )
                
                with tab_action1:
                    cap1_df = anomaly_df[anomaly_df['Mức độ rủi ro'] == "🔴 NGUY HIỂM CẤP 1"]
                    if len(cap1_df) > 0:
                        st.error(f"**⚠️ Phát hiện {len(cap1_df)} hồ sơ ở mức NGUY HIỂM CẤP 1**")
                        st.markdown("""
                        ### 🚨 Hành động khẩn cấp:
                        1. **Tạm dừng ngay lập tức** - Khoá các giao dịch liên quan
                        2. **Báo cáo cao cấp** - Thông báo cho lãnh đạo/kiểm toán
                        3. **Điều tra chi tiết** - Kiểm tra lịch sử giao dịch toàn bộ
                        4. **Bảo tồn bằng chứng** - Thu thập và lưu giữ tất cả dữ liệu
                        5. **Thông báo cơ quan** - Báo cáo nếu cần thiết theo quy định
                        
                        **Tiêu chí nhận dạng:** Risk Score ≥ 0.8 (Mức độ bất thường rất cao)
                        """)
                        st.dataframe(cap1_df[['risk_score'] + numeric_cols[:5]], use_container_width=True)
                    else:
                        st.success("✅ Không có hồ sơ ở mức Nguy hiểm cấp 1")
                
                with tab_action2:
                    cap2_df = anomaly_df[anomaly_df['Mức độ rủi ro'] == "🟠 NGUY HIỂM CẤP 2"]
                    if len(cap2_df) > 0:
                        st.warning(f"**⚠️ Phát hiện {len(cap2_df)} hồ sơ ở mức NGUY HIỂM CẤP 2**")
                        st.markdown("""
                        ### ⚠️ Hành động ưu tiên:
                        1. **Kiểm tra trong 24 giờ** - Rà soát kỹ lưỡng hồ sơ
                        2. **Liên hệ người liên quan** - Xác minh thông tin giao dịch
                        3. **So sánh dữ liệu** - Đối chiếu với hệ thống bên hợp tác
                        4. **Ghi nhận chi tiết** - Lưu lại các dấu hiệu bất thường
                        5. **Quyết định hành động** - Chấp nhận/từ chối/tạm dừng
                        
                        **Tiêu chí nhận dạng:** Risk Score 0.6-0.8 (Bất thường đáng kể)
                        """)
                        st.dataframe(cap2_df[['risk_score'] + numeric_cols[:5]], use_container_width=True)
                    else:
                        st.success("✅ Không có hồ sơ ở mức Nguy hiểm cấp 2")
                
                with tab_action3:
                    cao_df = anomaly_df[anomaly_df['Mức độ rủi ro'].isin(["🟡 RỦI RO CAO", "🟢 RỦI RO TRUNG BÌNH"])]
                    if len(cao_df) > 0:
                        st.info(f"**ℹ️ Phát hiện {len(cao_df)} hồ sơ ở mức RỦI RO CAO/TRUNG BÌNH**")
                        st.markdown("""
                        ### 📌 Hành động theo dõi:
                        1. **Giám sát thường xuyên** - Theo dõi hàng tuần
                        2. **Phân tích thêm** - Xem xét ngữ cảnh kinh doanh
                        3. **Yêu cầu giải thích** - Hỏi về lý do bất thường
                        4. **Tăng cường kiểm tra** - Rà soát chi tiết hơn
                        5. **Lên kế hoạch** - Xem có cần hành động hay không
                        
                        **Tiêu chí nhận dạng:** Risk Score 0.4-0.6 (Bất thường vừa phải)
                        """)
                        st.dataframe(cao_df[['Mức độ rủi ro', 'risk_score'] + numeric_cols[:5]], use_container_width=True)
                    else:
                        st.success("✅ Không có hồ sơ ở mức Rủi ro cao")
                
                with tab_action4:
                    st.markdown("""
                    ### 📊 Tóm tắt kết quả rà soát dữ liệu:
                    
                    #### **KẾT QUẢ TỔNG THỂ:**
                    """)
                    
                    cap1_count = len(anomaly_df[anomaly_df['Mức độ rủi ro'] == "🔴 NGUY HIỂM CẤP 1"])
                    cap2_count = len(anomaly_df[anomaly_df['Mức độ rủi ro'] == "🟠 NGUY HIỂM CẤP 2"])
                    cao_count = len(anomaly_df[anomaly_df['Mức độ rủi ro'].isin(["🟡 RỦI RO CAO", "🟢 RỦI RO TRUNG BÌNH"])])
                    
                    summary_text = f"""
                    - **Tổng hồ sơ nghi vấn:** {len(anomaly_df)} / {len(df)} ({risk_pct:.2f}%)
                    - 🔴 **Nguy hiểm cấp 1:** {cap1_count} hồ sơ (cần xử lý ngay)
                    - 🟠 **Nguy hiểm cấp 2:** {cap2_count} hồ sơ (cần xử lý trong 24h)
                    - 🟡🟢 **Rủi ro cao/TBình:** {cao_count} hồ sơ (giám sát thường xuyên)
                    
                    ---
                    
                    #### **ĐỀ XUẤT CHUNG:**
                    
                    1. **Ưu tiên cao nhất**: Xử lý ngay các hồ sơ cấp 1 (nếu có)
                    2. **Ưu tiên thứ 2**: Kiểm tra trong 24h các hồ sơ cấp 2
                    3. **Ưu tiên thứ 3**: Giám sát định kỳ các hồ sơ rủi ro cao
                    4. **Phòng chống dài hạn**: 
                       - Cập nhật quy trình kiểm soát nội bộ
                       - Đào tạo nhân viên về dấu hiệu gian lận
                       - Nâng cấp hệ thống AI phát hiện bất thường
                    5. **Báo cáo quản lý**: Tóm tắt hàng tuần/tháng
                    
                    ---
                    
                    **Ngày rà soát:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}
                    **Trạng thái:** ✅ Rà soát hoàn tất
                    """
                    st.markdown(summary_text)
            
            else:
                st.success("""
                ✅ **KẾT QUẢ: KHÔNG phát hiện hồ sơ nghi vấn**
                
                Dữ liệu hiện tại trong tình trạng tốt, không có dấu hiệu gian lận.
                
                **Đề xuất:**
                - Tiếp tục giám sát thường xuyên
                - Rà soát định kỳ (tháng/quý)
                - Cập nhật mô hình phát hiện khi có dữ liệu mới
                """)
    
    except Exception as e:
        st.error(f"❌ Lỗi xử lý file: {str(e)}")

else:
    st.info("Vui lòng tải file CSV lên để bắt đầu phân tích.")