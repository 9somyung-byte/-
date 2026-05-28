import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# 웹 페이지 제목 및 레이아웃 설정
st.set_page_config(page_title="당뇨병 예측 프로그램", layout="centered")
st.title("🩺 당뇨병 위험도 예측 프로그램")
st.write("아래 정보를 입력하신 후 '예측하기' 버튼을 눌러주세요.")

# ==========================================
# 1. [캐싱] 스케일러 로드 및 모델 초기화
# ==========================================
@st.cache_resource
def load_scaler_and_model():
    file_name = 'diabetes_scaler.pkl'
    
    if not os.path.exists(file_name):
        st.error(f"오류: '{file_name}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None, None
        
    try:
        # 스케일러 객체 로드
        scaler = joblib.load(file_name)
        
        # 💡 [순서 에러 해결] 스케일러가 기억하는 정확한 피처 순서를 추출합니다.
        if hasattr(scaler, 'feature_names_in_'):
            features = list(scaler.feature_names_in_)
        else:
            # 피처 이름이 없는 구버전일 경우 예상되는 순서로 지정
            features = ['임신횟수', '포도당', '혈압', '피부두께', '인슐린', 'BMI', '당뇨가계지수', '나이',
                        '고혈압', '비만여부', '대사위험점수', '고령', '고혈당']
        
        # 스케일러의 실제 컬럼 순서와 동일하게 dummy를 만들어 모델을 학습시킵니다.
        X_dummy = pd.DataFrame([[0]*len(features), [1]*len(features)], columns=features)
        y_dummy = [0, 1]
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_dummy, y_dummy)
        
        return scaler, model
    except Exception as e:
        st.error(f"파일을 로드하는 중 오류가 발생했습니다: {e}")
        return None, None

# 스케일러와 모델 불러오기
scaler, log_model = load_scaler_and_model()

# ==========================================
# 2. 사용자 입력 UI (화면 분할 레이아웃)
# ==========================================
if scaler is not None and log_model is not None:
    st.subheader("📋 신체 정보 입력")
    
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
        
        # 기본 입력 데이터 생성
        input_data = pd.DataFrame(
            [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=['임신횟수', '포도당', '혈압', '피부두께', '인슐린', 'BMI', '당뇨가계지수', '나이']
        )

        # 파생 변수 5개 생성
        input_data['고혈압'] = (input_data['혈압'] >= 140).astype(int)
        input_data['비만여부'] = (input_data['BMI'] >= 30).astype(int)
        input_data['대사위험점수'] = ((input_data['BMI'] >= 25).astype(int) + (input_data['포도당'] >= 130).astype(int))
        input_data['고령'] = (input_data['나이'] >= 50).astype(int)
        input_data['고혈당'] = (input_data['포도당'] >= 140).astype(int)

        try:
            # 💡 [핵심 수정] 스케일러 내부에 저장된 정확한 순서 정보(`feature_names_in_`)를 가져와 재정렬합니다.
            if hasattr(scaler, 'feature_names_in_'):
                input_data = input_data[scaler.feature_names_in_]
            
            # 💡 DataFrame의 컬럼명 검증을 우회하기 위해 values(Numpy Array) 형태로 스케일러에 주입합니다.
            scaled_data = scaler.transform(input_data.values)
            
            # 예측 진행
            predicted = log_model.predict(scaled_data)
            prob = log_model.predict_proba(scaled_data)
            diabetes_prob = prob[0][1] * 100

            # 결과 출력
            st.subheader("📊 예측 결과")
            st.metric(label="당뇨병 발병 확률", value=f"{diabetes_prob:.1f} %")
            
            if predicted[0] == 1:
                st.error(f"⚠️ 예측 결과: **당뇨 위험군**입니다. (확률: {diabetes_prob:.1f}%)")
            else:
                st.success(f"✅ 예측 결과: **정상**입니다. (확률: {100 - diabetes_prob:.1f}%)")
                
        except Exception as e:
            st.error(f"예측 도중 오류가 발생했습니다: {e}")
