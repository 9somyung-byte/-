import streamlit as st
import numpy as np
import joblib

# =========================
# 모델 & 스케일러 로드
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("diabetes_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =========================
# 제목
# =========================
st.title("당뇨병 예측 앱")

st.write("값을 입력하면 당뇨병 여부를 예측합니다.")

# =========================
# 사용자 입력
# =========================
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# =========================
# 예측 버튼
# =========================
if st.button("예측하기"):

    # 입력 데이터를 배열로 변환
    input_data = np.array([
        [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]
    ])

    # 스케일링
    input_data_scaled = scaler.transform(input_data)

    # 예측
    prediction = model.predict(input_data_scaled)

    # 결과 출력
    if prediction[0] == 1:
        st.error("당뇨병일 가능성이 높습니다.")
    else:
        st.success("당뇨병일 가능성이 낮습니다.")
