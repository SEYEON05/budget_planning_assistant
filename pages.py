import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
import numpy as np
import time

# 데이터 가오져는 함수
@st.cache_data
def get_datas(name):
    data = pd.read_csv(name, index_col=0)
    return data
data = get_datas("student_spending (1).csv")
analyzed_data = get_datas("profit_spending.csv")

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
  fixed_spending_cols = ['housing', 'personal_care', 'technology', 'health_wellness']
  variable_spending_col = ['food', 'transportation', 'books_supplies', 'entertainment', 'miscellaneous']
  tmp = tmp[tmp.columns].drop(columns=['tuition'])


  # 고정지출과 변동지출 그래프
  st.divider()
  with st.expander("**고정지출과 변동지출 그래프**"):
    st.text("대학생 1000명의 총 고정지출과 총 변동지출을 나타낸 그래프이다.")
    
    tmp_spending = pd.DataFrame(analyzed_data, columns=['total_fixed_spending', 'total_variable_spending'])
    mean_fixed_spending = analyzed_data['total_fixed_spending'].mean()
    mean_variable_spending = analyzed_data['total_variable_spending'].mean()
    mean_total_spending = analyzed_data['total_spending'].mean()
    with st.container(border=True):
      st.caption('단위: 달러')
      st.line_chart(tmp_spending)
    st.info(f"""총 고정지출(total_fixed_spending)은 각 학생들에 대한
{fixed_spending_cols}의 합이다.

총 변동지출(total_variable_spending)은 각 학생들에 대한
{variable_spending_col}의 합이다.

총 고정지출은 대체로 총 변동지출보다 많다. 학생들의 평균 고정지출은 {mean_fixed_spending} 달러이고, 평균 변동지출은 {mean_variable_spending} 달러이다.
따라서, 학생들의 한달 평균 지출은 {mean_total_spending} 달러임을 확인할 수 있다.""")


  # 셀렉트박스 중 하나를 선택 시, 스탠다드를 기준으로 평균치가 계산되게

  st.subheader("기본 정보에 따른 소비 형태")
  standard = st.selectbox('항목을 선택하세요.', ("Age", "Gender", "Year_in_School", "Major", "Preferred_Payment_Method", "Classified_Profit")).lower()
  standard_cols = ['age', 'gender', 'year_in_school', 'major', 'preferred_payment_method', 'monthly_income', 'financial_aid', 'classified_profit']
  
  if standard in standard_cols:
    cols_to_drop = [col for col in standard_cols if col != standard]
    tmp = tmp[tmp.columns].drop(columns=cols_to_drop)


  # 기본 정보에 따른 지출 항목 그래프
  with st.expander(f"**{standard} 에 따른 지출 항목 그래프**"):
    st.markdown(f"**{standard} 에 따른 지출 항목 그래프**")
    tmp_mean = tmp.groupby(standard).mean()
    tmp_mean_T = tmp_mean.T
    with st.container(border=True):
      st.line_chart(tmp_mean_T, height=600)

  # 수입에 따른 저축 그래프
  with st.expander("**수입과 저축 분석**"):
    # st.markdown("수입과 저축 그래프")
    # with st.container(border=True):
    #   st.area_chart(analyzed_data, x='total_profit', y='savings')
    
    # 두 개의 기본 정보에 따른 변수 그래프
    sns.set_theme(style="whitegrid")
    h = sns.catplot(
      data=analyzed_data, kind='bar',
      x='classified_profit', y='savings', hue=standard,
      errorbar='sd', palette='dark', alpha=.6, height=6
    )
    h.despine(left=True)
    h.set_axis_labels("classified_profit", "savings")
    st.markdown("수입 정도에 따른 저축 그래프")
    with st.container(border=True):
      st.pyplot(h)

  # 기본 정보에 따른 수입과 지출
  with st.expander(f"**{standard} 에 따른 수입과 지출 분석**"):
    st.markdown(f"**{standard} 에 따른 수입과 지출 분석**")
    sns.set_theme()
    g = sns.lmplot(
      data=analyzed_data,
      x="total_profit", y="total_spending", hue=standard,
      height=7
    )
    g.set_axis_labels("total_profit", "total_spending")
    with st.container(border=True):  
      st.pyplot(g)
      if standard == 'preferred_payment_method':
        st.info("선호 지출 방식에 따른 수입-지출 분석 결과")
        st.markdown("카드를 사용하는 사용자가 가장 많은 지출을 하는 것으로 분석된다.")

  # 기본 정보에 따른 수입과 고정지출
  with st.expander(f"**{standard}에 따른 총수익-고정지출 그래프**"):
    st.markdown(f"**{standard}에 따른 총수익-고정지출 그래프**")
    sns.set_theme()
    g = sns.lmplot(
      data=analyzed_data,
      x="total_profit", y="total_fixed_spending", hue=standard,
      height=7
    )
    g.set_axis_labels("total_profit", "total_fixed_spending")
    with st.container(border=True):  
      st.pyplot(g)

# 기본 정보에 따른 수입과 변동지출
  with st.expander("**기본 정보에 따른 총수익-변동지출 그래프**"):
    st.markdown("기본 정보에 따른 총수익-변동지출 그래프")
    sns.set_theme()
    g = sns.lmplot(
      data=analyzed_data,
      x="total_profit", y="total_variable_spending", hue=standard,
      height=7
    )
    g.set_axis_labels("total_profit", "total_variable_spending")
    with st.container(border=True):  
      st.pyplot(g)



# 예산 계획 도우미 페이지
def planner():
  st.subheader('예산 계획 도우미')
  my_income = st.number_input("이번 달 예상 수입을 입력하세요.")
  my_savings = st.number_input("이번 달 저축액을 입력하세요.")
  my_fixed = st.number_input("나의 고정지출을 입력하세요.")
  available = my_income - my_savings
  available_fixed = available * 0.58
  available_variable = available - my_fixed
  
  
  # 아래 결과화면은 버튼을 누르면 실행되게
  
  if st.button("실행", type="primary", use_container_width=True) == False:
    st.markdown("")
  elif my_fixed > available_fixed:
    st.text(f"고정지출이 평균보다 {my_fixed - available_fixed}달러 많아요. 줄여야할 필요가 있어요!")
    st.text(f"변동지출 가능액: {available_variable}")
  else:
    st.text("고정지출이 적당해요. 잘하고 있어요!")
    st.text(f"변동지출 가능액: {available_variable}")
  result = pd.DataFrame()


  #   # 이전에 데이터프레임 만들고 파일에 입력
  #   # with open("result.csv", "w") as f:
  #   #   f.write(# 데이터프레임)  


