import streamlit as st
import numpy as np
import joblib

# =========================
# 모델 & 스케일러 불러오기
# =========================
@st.cache_resource
def 모델불러오기():
    모델 = joblib.load("diabetes_model.pkl")
    스케일러 = joblib.load("diabetes_scaler.pkl")
    return 모델, 스케일러

모델, 스케일러 = 모델불러오기()

# =========================
# 제목
# =========================
st.title("🩺 당뇨병 예측 프로그램")

st.write("건강 정보를 입력하면 당뇨병 가능성을 예측합니다.")

# =========================
# 현재 모델 정보 출력
# =========================
st.subheader("📌 모델 정보")

st.write("필요 입력 컬럼 개수:", 스케일러.n_features_in_)

if hasattr(스케일러, "feature_names_in_"):
    st.write("학습된 컬럼 이름:")
    st.write(list(스케일러.feature_names_in_))

# =========================
# 사용자 입력
# =========================
st.subheader("📋 건강 정보 입력")

임신횟수 = st.number_input("임신 횟수", min_value=0)

포도당 = st.number_input("포도당 수치", min_value=0)

혈압 = st.number_input("혈압", min_value=0)

피부두께 = st.number_input("피부 두께", min_value=0)

인슐린 = st.number_input("인슐린 수치", min_value=0)

BMI = st.number_input("BMI", min_value=0.0)

당뇨유전확률 = st.number_input(
    "당뇨 유전 확률",
    min_value=0.0
)

나이 = st.number_input("나이", min_value=0)

# =========================
# 예측 버튼
# =========================
if st.button("예측하기"):

    try:

        # 현재 입력 데이터 (8개)
        입력데이터 = [
            임신횟수,
            포도당,
            혈압,
            피부두께,
            인슐린,
            BMI,
            당뇨유전확률,
            나이
        ]

        # =========================
        # 부족한 컬럼 자동 추가
        # =========================
        필요한컬럼수 = 스케일러.n_features_in_

        현재컬럼수 = len(입력데이터)

        부족한컬럼수 = 필요한컬럼수 - 현재컬럼수

        # 부족한 만큼 0 추가
        if 부족한컬럼수 > 0:
            입력데이터.extend([0] * 부족한컬럼수)

        # numpy 배열 변환
        입력데이터 = np.array([입력데이터])

        # =========================
        # 스케일링
        # =========================
        입력데이터스케일 = 스케일러.transform(입력데이터)

        # =========================
        # 예측
        # =========================
        예측결과 = 모델.predict(입력데이터스케일)

        예측확률 = 모델.predict_proba(입력데이터스케일)

        # =========================
        # 결과 출력
        # =========================
        st.subheader("📊 예측 결과")

        if 예측결과[0] == 1:
            st.error("⚠️ 당뇨병일 가능성이 높습니다.")
        else:
            st.success("✅ 당뇨병 가능성이 낮습니다.")

        # 확률 출력
        당뇨확률 = 예측확률[0][1] * 100

        st.write(f"당뇨병 확률: {당뇨확률:.2f}%")

    except Exception as 오류:

        st.error("에러 발생")

        st.code(str(오류))
