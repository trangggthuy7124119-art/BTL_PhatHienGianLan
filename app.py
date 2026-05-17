import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models import load_data, prepare_data, train_and_evaluate, get_feature_importance, TARGET_COL
from styles import CSS_STYLE, PLOTLY_LAYOUT, COLORS

st.set_page_config(page_title='AI AML & Fraud Detection', layout='wide', page_icon='🛡️')
st.markdown(CSS_STYLE, unsafe_allow_html=True)

with st.sidebar:
    st.title('HỆ THỐNG AI AML & GIAN LẬN')
    page = st.radio('Chọn trang', ['🏠 Tổng quan', '📊 Phân tích', '🔍 Kết luận & Dự đoán'])
    test_size = st.slider('Tỷ lệ Test size', 0.1, 0.4, 0.2, 0.05)

@st.cache_data
def load_cached_data():
    return load_data()

def read_batch_file(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

try:
    df, feature_cols, target_col = load_cached_data()
except Exception as e:
    st.error(f'Không thể đọc dữ liệu: {e}')
    st.stop()

X_train, X_test, y_train, y_test = prepare_data(df, feature_cols, test_size)

if page == '🏠 Tổng quan':
    st.header('Tổng quan AML & Fraud Patterns')
    total = len(df)
    counts = df[target_col].value_counts().sort_index()
    labels = ['Normal', 'Default']
    values = [counts.get(i, 0) for i in range(len(labels))]
    percents = [f'{v/total*100:.1f}%' for v in values]

    col1, col2 = st.columns(2)
    col1.markdown(f"<div class='metric-card'><h3>{total}</h3><p>Tổng số mẫu</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>{values[0]}</h3><p>Bình thường ({percents[0]})</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card'><h3>{values[1]}</h3><p>Default ({percents[1]})</p></div>", unsafe_allow_html=True)

    fig_pie = px.pie(
        names=labels, values=values,
        title='Phân bố trạng thái mục tiêu', color=labels,
        color_discrete_sequence=COLORS,
        hole=0.35
    )
    fig_pie.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader('Thống kê mô tả các chỉ số tài chính')
    st.dataframe(df[feature_cols].describe().T.style.background_gradient(cmap='viridis'))

    st.subheader('Ma trận tương quan giữa các chỉ số')
    corr = df[feature_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='blues')
    fig_corr.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_corr, use_container_width=True)

elif page == '📊 Phân tích':
    st.header('So sánh mô hình AI cho AML & Fraud')
    if st.button('🚀 Chạy huấn luyện hệ thống') or 'trained' not in st.session_state:
        st.session_state['trained'] = train_and_evaluate(X_train, X_test, y_train, y_test)
    results = st.session_state['trained']

    metrics = []
    for name, data in results.items():
        metrics.append({
            'Model': name,
            'Accuracy': round(data['accuracy'], 3),
            'Precision': round(data['precision'], 3),
            'Recall': round(data['recall'], 3),
            'F1-Score': round(data['f1'], 3),
        })
    df_metrics = pd.DataFrame(metrics)
    st.subheader('Bảng so sánh hiệu suất')
    st.dataframe(df_metrics.set_index('Model'))

    fig_bar = px.bar(
        df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score'),
        x='Model', y='Score', color='Metric', barmode='group', color_discrete_sequence=COLORS
    )
    fig_bar.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader('Ma trận nhầm lẫn của từng mô hình')
    for name, data in results.items():
        fig_cm = px.imshow(data['confusion_matrix'], text_auto=True, color_continuous_scale='Reds')
        fig_cm.update_layout(title=name, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_cm, use_container_width=True)

    importance_frames = []
    for name, data in results.items():
        importance = get_feature_importance(data['model'], feature_cols)
        importance['Model'] = name
        importance_frames.append(importance)
    importances = pd.concat(importance_frames)
    fig_imp = px.bar(
        importances, x='importance', y='feature', color='Model', orientation='h', height=600,
        color_discrete_sequence=COLORS
    )
    fig_imp.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.header('Kết luận & Dự đoán cá nhân hóa')
    tab1, tab2 = st.tabs(['✍️ Nhập thông tin thủ công', '📁 Tải file phân tích hàng loạt'])

    with tab1:
        st.write('Nhập các chỉ số tài chính cho một tài khoản mới và chọn mô hình dự đoán.')
        with st.form('predict_form'):
            inputs = {}
            for col in feature_cols:
                inputs[col] = st.number_input(col, value=float(df[col].median()), step=0.1)
            model_name = st.selectbox('Chọn mô hình', ['Random Forest', 'Logistic Regression', 'XGBoost'])
            submit = st.form_submit_button('🔍 Phân tích & Đưa ra kết luận')

        if submit:
            results = st.session_state.get('trained') or train_and_evaluate(X_train, X_test, y_train, y_test)
            model = results[model_name]['model']
            sample = pd.DataFrame([inputs])[feature_cols]
            proba = model.predict_proba(sample)[0]
            label = model.predict(sample)[0]
            labels = ['Normal', 'Default']
            confidence = proba[label] * 100
            risk_class = 'result-normal'
            message = 'Dự đoán trạng thái Bình thường.'
            if label == 1:
                risk_class = 'result-fraud'
                message = 'Cảnh báo khoản vay có nguy cơ vỡ nợ. Cần điều tra thêm.'

            st.markdown(f"<div class='result-box {risk_class}'>"
                        f"<h3>{labels[label]}</h3>"
                        f"<p>{message}</p>"
                        f"<p><strong>Độ tin cậy:</strong> {confidence:.1f}%</p>"
                        f"</div>", unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                'Trạng thái': labels,
                'Xác suất': [f'{p*100:.1f}%' for p in proba]
            })
            st.table(prob_df)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=list(inputs.values()),
                theta=list(inputs.keys()),
                fill='toself', name='Giá trị nhập'
            ))
            fig_radar.update_layout(
                polar={'bgcolor': '#0e1117'},
                **PLOTLY_LAYOUT,
                showlegend=False,
                title='Radar chart các chỉ số đầu vào'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.write('Upload file dữ liệu để hệ thống phân tích hàng loạt.')
        upload_file = st.file_uploader(
            'Chọn file .csv, .xls hoặc .xlsx',
            type=['csv', 'xls', 'xlsx'],
            help='File đầu vào cần có các cột dữ liệu giống bộ tính năng hiện tại.'
        )
        batch_model_name = st.selectbox('Chọn mô hình hàng loạt', ['Random Forest', 'Logistic Regression', 'XGBoost'])
        batch_button = st.button('🚀 Tiến hành quét dữ liệu hàng loạt')

        if batch_button:
            if upload_file is None:
                st.warning('Vui lòng upload tệp dữ liệu trước khi quét.')
            else:
                batch_df = read_batch_file(upload_file)
                missing = [c for c in feature_cols if c not in batch_df.columns]
                if missing:
                    st.error(f'Tệp thiếu cột bắt buộc: {missing}')
                else:
                    results = st.session_state.get('trained') or train_and_evaluate(X_train, X_test, y_train, y_test)
                    model = results[batch_model_name]['model']
                    batch_X = batch_df[feature_cols].copy()
                    batch_pred = model.predict(batch_X)
                    batch_proba = model.predict_proba(batch_X)
                    batch_df['Dự_đoán_Kết_luận'] = ['Normal' if p == 0 else 'Anomaly' for p in batch_pred]
                    batch_df['Độ_tin_cậy_AI (%)'] = [float(max(p) * 100) for p in batch_proba]

                    count_normal = (batch_df['Dự_đoán_Kết_luận'] == 'Normal').sum()
                    count_anomaly = (batch_df['Dự_đoán_Kết_luận'] != 'Normal').sum()
                    st.subheader('Thống kê nhanh')
                    st.metric('Normal', count_normal)
                    st.metric('Anomaly / Risk', count_anomaly)

                    st.subheader('10 dòng đầu tiên có dự đoán')
                    st.dataframe(batch_df.head(10))

                    towrite = io.BytesIO()
                    batch_df.to_excel(towrite, index=False, engine='openpyxl')
                    towrite.seek(0)
                    st.download_button(
                        'Tải file kết quả Excel',
                        data=towrite,
                        file_name='batch_prediction_result.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
