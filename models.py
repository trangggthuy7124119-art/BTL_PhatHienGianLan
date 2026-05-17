import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

TARGET_COL = 'default payment next month'

@st.cache_data
def load_data():
    """Đọc dữ liệu từ file Excel và chuẩn hóa nhãn mục tiêu."""
    df = pd.read_excel('data.xls', engine='xlrd', header=1)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{TARGET_COL}' trong data.xls")

    df = df.copy()
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
        label_map = {
            'Normal': 0,
            'normal': 0,
            'Fraud': 1,
            'fraud': 1,
            'High Risk': 2,
            'HighRisk': 2,
            'high risk': 2,
            'Risk': 2,
        }
        df[TARGET_COL] = df[TARGET_COL].map(label_map).fillna(df[TARGET_COL])
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL], _ = pd.factorize(df[TARGET_COL])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in [TARGET_COL, 'ID']]
    if len(feature_cols) > 13:
        feature_cols = feature_cols[:13]
    df = df[[TARGET_COL] + feature_cols]
    return df, feature_cols, TARGET_COL

@st.cache_data
def prepare_data(df, feature_cols, test_size):
    """Chia dữ liệu train/test với stratify theo nhãn để giữ tỷ lệ class."""
    X = df[feature_cols]
    y = df[TARGET_COL]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

@st.cache_resource
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Huấn luyện ba mô hình và trả về kết quả đánh giá."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear'),
    }
    if XGBClassifier is None:
        raise ImportError('Vui lòng cài xgboost để sử dụng XGBoostClassifier')
    models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    weights = compute_sample_weight('balanced', y_train)
    results = {}
    for name, model in models.items():
        if name == 'XGBoost':
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predict_proba': y_prob,
        }
    return results

def get_feature_importance(model, feature_cols):
    """Lấy độ quan trọng của biến từ mô hình đã huấn luyện."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        importances = np.zeros(len(feature_cols))
    return pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
