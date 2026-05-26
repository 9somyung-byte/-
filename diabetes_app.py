import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# 웹 페이지 제목 및 레이아웃 설정
st.set_page_config(page_title="당뇨병 예측 프로그램", layout="centered")
st.title("🩺 당뇨병 위험도 예측 프로그램")
st.write("아래 정보를 입력하신 후 '예측하기' 버튼을 눌러주세요.")

# ==========================================
# 1. [캐싱] 데이터 로드 및 모델 학습 함수
# ==========================================
@st.cache_resource
def load_model():
    try:
        # pkl 파일이 데이터프레임(DataFrame)인 경우
        df = pd.read_pickle('diabetes_scaler.pkl') 
        
        # 파생 변수 생성
        df['비만위험'] = (df['BMI'] >= 30).astype(int)
        df['고혈당'] = (df['포도당'] >= 140).astype(int)
        df['고령'] = (df['나이'] >= 50).astype(int)
        df['대사위험'] = ((df['BMI'] >= 25).astype(int) + (df['포도당'] >= 130).astype(int))

        # 독립변수와 종속변수 분리
        X = df[['임신횟수','포도당','혈압','피부두께','인슐린','BMI','당뇨가계지수','나이',
                '비만위험','고혈당','고령','대사위험']]
        y = df['결과']

        # 모델 학습
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        return model
    except FileNotFoundError:
        st.error("오류: 'diabetes_model.pkl' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None

# 모델 불러오기
log_model = load_model()

# ==========================================
# 2. 사용자 입력 UI (화면 분할 레이아웃)
# ==========================================
if log_model is not None:
    st.subheader("📋 신체 정보 입력")
    
    # 화면을 좌우 2열로 나눕니다.
    col1, col2 = st.columns(2)
    
    with col1:
        preg = st.number_input("임신 횟수", min_value=0, max_value=20, value=0, step=1)
        glucose = st.number_input("포도당 수치 (mg/dL)", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
        bp = st.number_input("혈압 (mmHg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        skin = st.number_input("피부 두께 (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        
    with col2:
        insulin = st.number_input("인슐린 수치 (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI (비만도 체질량지수)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("당뇨 가계 지수", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.3f")
        age = st.number_input("나이", min_value=1, max_value=120, value=30, step=1)

    st.markdown("---")

    # ==========================================
    # 3. 예측 실행 및 결과 출력
    # ==========================================
    if st.button("🔍 당뇨 위험도 예측하기", type="primary", use_container_width=True):
        
        # 입력받은 데이터로 DataFrame 생성
        input_data = pd.DataFrame(
            [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=['임신횟수', '포도당', '혈압', '피부두께', '인슐린', 'BMI', '당뇨가계지수', '나이']
        )

        # 파생 변수 자동 계산
        input_data['비만위험'] = (input_data['BMI'] >= 30).astype(int)
        input_data['고혈당'] = (input_data['포도당'] >= 140).astype(int)
        input_data['고령'] = (input_data['나이'] >= 50).astype(int)
        input_data['대사위험'] = ((input_data['BMI'] >= 25).astype(int) + (input_data['포도당'] >= 130).astype(int))

        # 예측 진행
        predicted = log_model.predict(input_data)
        prob = log_model.predict_proba(input_data)
        diabetes_prob = prob[0][1] * 100

        # 결과 출력 레이아웃
        st.subheader("📊 예측 결과")
        
        # 메트릭 카드로 확률 보여주기
        st.metric(label="당뇨병 발병 확률", value=f"{diabetes_prob:.1f} %")
        
        if predicted[0] == 1:
            st.error(f"⚠️ 예측 결과: **당뇨 위험군**입니다. (확률: {diabetes_prob:.1f}%)")
        else:
            st.success(f"✅ 예측 결과: **정상**입니다. (확률: {100 - diabetes_prob:.1f}%)")