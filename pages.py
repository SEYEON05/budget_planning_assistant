import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
import numpy as np
import time

# 데이터 가오져는 함수
@st.cache_data
def get_datas():
    data = pd.read_csv("student_spending (1).csv", index_col=0)
    return data
data = get_datas()

def txt_gen(txt):
  for t in list(txt):
    yield t
    time.sleep(0.01)

# home 페이지
def home():
  HELLO_1 = "저는 B.PA라고 해요! 'Budget Planning Assistant'에서 따와 지은 이름이죠. :wink:"
  HELLO_2 = "대학생이 되어 요즘 부쩍 바빠진 당신을 위해 제가 한달 소비 계획을 도와드릴게요."
  
  st.header("Hello! I am your B.PA!")
  # st.markdown('저는 B.PA라고 해요! "Budget Planning Assistant"에서 따와 지은 이름이죠. :wink:')
  # st.markdown('대학생이 되어 요즘 부쩍 바빠진 당신을 위해 제가 한달 소비 계획을 도와드릴게요.')
  st.write_stream(txt_gen(HELLO_1))
  st.write_stream(txt_gen(HELLO_2))


  st.divider()
  st.info(":information_source: Information")
  st.markdown("1. 제가 사용하는 데이터는 아래와 같아요.")
  st.caption("(단위: 달러)")
  st.write(data.head())
  st.markdown("2. 1000명의 대학생들의 소비 행태 데이터를 사용합니다.")
  st.markdown("3. **소비 패턴 분석**: Pandas와 Streamlit을 활용해 소비 패턴 분석 및 시각화를 효과적으로 전달합니다.")
  st.markdown("4. **예산 계획 도우미**: 사용자 입력(월수입, 저축액)을 받아 합리적 예산 분배를 제안합니다.")
# 소비 습관 분석 페이지
def pattern_analyis():
  st.title('소비 패턴 분석')
  # 데이터 카피
  tmp = data.copy()

  # 숫자형으로 변환할 수 있는 컬럼을 숫자형으로 변환, 데이터 값의 type object 이슈 발생했음
  cols_to_convert = ['monthly_income', 'financial_aid', 'tuition', 'housing', 'food', 'transportation', 'books_supplies', 'entertainment', 'personal_care', 'technology', 'health_wellness', 'miscellaneous']
  for col in cols_to_convert:
    tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
  
  # 안 쓰는 컬럼 ['tuition', 'monthly_income', 'financial_aid'] 제거
  tmp = tmp[tmp.columns].drop(columns=['tuition', 'monthly_income', 'financial_aid'])
  
  # 셀렉트박스 중 하나를 선택 시, 스탠다드를 기준으로 평균치가 계산되게
  # standard = st.selectbox('항목을 선택하세요.', ("age", "gender", "year_in_school", "major", "preferred_payment_method"))
  standard = st.selectbox('항목을 선택하세요.', ("Age", "Gender", "Year_in_School", "Major", "Preferred_Payment_Method")).lower()
  
  standard_cols = ['age', 'gender', 'year_in_school', 'major', 'preferred_payment_method']
  
  if standard in standard_cols:
    cols_to_drop = [col for col in standard_cols if col != standard]
    tmp = tmp[tmp.columns].drop(columns=cols_to_drop)
    st.divider()
    st.markdown(f"**{standard} 에 따른 소비 형태**")
  # st.write(tmp)
  tmp_mean = tmp.groupby(standard).mean()
  tmp_mean_T = tmp_mean.T
  with st.container(border=True):
    st.line_chart(tmp_mean_T, height=600)

    # seaborn을 사용한 그래프 그리기 시도, 결과: 별로
    # fig, ax = plt.subplots()
    # sns.lineplot(data=tmp_mean_T)
    # ax.set_title(f"**{standard} 에 따른 소비 형태**")
    # st.pyplot(fig)


# 예산 계획 도우미 페이지
def planner():
  st.subheader('예산 계획 도우미')
  income = st.number_input("이번 달 예상 수입을 입력하세요.")
  saving = st.number_input("이번 달 저축액을 입력하세요.")
  available = income - saving