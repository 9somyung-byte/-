import streamlit as st
import numpy as np
import joblib

# =========================
# 모델 & 스케일러 로드
# =========================
@st.cache_resource
def 모델불러오기():
    모델 = joblib.load("diabetes_model.pkl")
    스케일러 = joblib.load("diabetes_scaler.pkl")
    return 모델, 스케일러

모델, 스케일러 = 모델불러오기()

# =========================
# 앱 제목
# =========================
st.title("🩺 당뇨병 예측 프로그램")

st.write("건강 정보를 입력하면 당뇨병 가능성을 예측합니다.")

# =========================
# 사용자 입력
# =========================
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

    입력데이터 = np.array([
        [
            임신횟수,
            포도당,
            혈압,
            피부두께,
            인슐린,
            BMI,
            당뇨유전확률,
            나이
        ]
    ])

    try:

        # scaler 입력 feature 개수 확인
        필요한컬럼수 = 스케일러.n_features_in_

        현재컬럼수 = 입력데이터.shape[1]

        if 필요한컬럼수 != 현재컬럼수:

            st.error(
                f"""
                입력 컬럼 개수가 다릅니다.

                현재 입력 개수: {현재컬럼수}
                필요한 개수: {필요한컬럼수}

                모델 학습 시 사용한 컬럼 수와
                현재 입력 컬럼 수가 다릅니다.
                """
            )

        else:

            # 스케일링
            입력데이터스케일 = 스케일러.transform(입력데이터)

            # 예측
            예측결과 = 모델.predict(입력데이터스케일)

            # 결과 출력
            if 예측결과[0] == 1:
                st.error("⚠️ 당뇨병일 가능성이 높습니다.")
            else:
                st.success("✅ 당뇨병 가능성이 낮습니다.")

    except Exception as 오류:
        st.error(f"에러 발생: {오류}")
