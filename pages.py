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
  
  # 안 쓰는 컬럼 'tuition' 제거
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
      errorbar='sd', palette='dark', alpha=.6, height=6,
      order=['under 1000', '1000-1500', '1500-2000', '2000-2500']
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
  dfp = data.copy()
  st.subheader('내 지출 수준 분석')
  # my_age = st.selectbox('나이를 선택하세요.', sorted(dfp['age'].unique()))
  # my_gender = st.selectbox('성별을 선택하세요.', dfp['gender'].unique())
  # my_major = st.selectbox('전공을 선택하세요.', dfp['major'].unique())

  my_yis = st.selectbox('학년을 선택하세요.', dfp['year_in_school'].unique())
  my_ppm = st.selectbox('선호하는 결제 방법을 선택하세요.', dfp['preferred_payment_method'].unique())
  my_income = st.number_input("이번 달 예상 수입을 입력하세요.")

  # 사용자로부터 선택 받을 옵션들
  options_list = ['집세', '식비', '교통비', '도서구입비', '여가비', '퍼스널케어', '통신비', '건강/운동', '잡비']

  # 옵션에 따른 입력 필드 변수명 매핑
  input_fields = {
      '집세': ['집세를 입력하세요.', 'housing'],
      '식비': ['식비를 입력하세요.', 'food'],
      '교통비': ['교통비를 입력하세요.', 'transportation'],
      '도서구입비': ['도서구입비를 입력하세요.', 'books_supplies'],
      '여가비': ['여가비를 입력하세요.', 'entertainment'],
      '퍼스널케어': ['퍼스널케어를 입력하세요.', 'personal_care'],
      '통신비': ['통신비를 입력하세요.', 'technology'],
      '건강/운동': ['건강/운동를 입력하세요.', 'health_wellness'],
      '잡비': ['잡비를 입력하세요.', 'miscellaneous']
  }

  # 사용자가 선택한 옵션
  options = st.multiselect(
      '다음 중 한달 지출액을 알고 있는 항목을 선택하세요.',
      options_list
  )

  # 사용자 입력 값을 저장하는 딕셔너리
  user_inputs = {}

  # 사용자가 선택한 옵션에 대해 입력 필드 생성하고, 입력 값을 user_inputs에 저장
  for option in options:
      if option in input_fields:
          user_inputs[input_fields[option][1]] = st.number_input(input_fields[option][0])

  
  # 조건문과 그에 해당하는 인덱스를 리스트로 가져오기
  condition = (dfp['year_in_school']==my_yis) & (dfp['preferred_payment_method']==my_ppm)
  matching_indexes = dfp.index[condition].tolist()

  selected_rows = dfp.loc[matching_indexes]
  
  # my_income에 따른 'classified_profit' 값 매핑을 위한 딕셔너리 생성
  income_to_profit = {
      range(0, 1000): 'under 1000',
      range(1000, 1500): '1000-1500',
      range(1500, 2000): '1500-2000',
      range(2000, 2501): '2000-2500'
  }

  # my_income에 해당하는 'classified_profit' 값을 찾기
  for income_range, profit_class in income_to_profit.items():
      if my_income in income_range:
          classified_profit_value = profit_class
          break

  # 찾은 'classified_profit' 값에 해당하는 행 선택
  selected_rows = selected_rows.loc[selected_rows['classified_profit'] == classified_profit_value]

  # 사용할 (value가 숫자인)columns만 남기기
  selected_rows = selected_rows.drop(columns=['age', 'gender', 'year_in_school', 'major', 'tuition', 'preferred_payment_method', 'classified_profit'])
  result = pd.DataFrame(data=[selected_rows.min(), selected_rows.mean(), selected_rows.max()], index=['min', 'mean', 'max'])

  if st.button("실행", type='primary',use_container_width=True):
    with st.expander("See explanation"):
      st.info("""이 프로그램은 사용자가 앞서 입력한 정보를 기반으로, 나와 비슷한 조건의 사용자 데이터를 골라내어 나의 지출수준을 분석한다.
              아래 슬라이더의 가장 왼쪽 값은 나와 비슷한 조건의 사용자들이 각 지출항목에 사용한 최솟값이고, 가장 오른쪽 값은 사용한 최댓값이다.""")
      st.select_slider('예시', options=['min', 'mean', 'max'], value=('mean'))
    col1, col2 = st.columns([5, 5])
    
    # 입력 값이 있는 변수와 없는 변수를 분류
    inputs_with_values = []
    inputs_without_values = []
    for cate in input_fields:
        if input_fields[cate][1] in user_inputs and user_inputs[input_fields[cate][1]] is not None:
            inputs_with_values.append(cate)
        else:
            inputs_without_values.append(cate)

    # 입력 값이 있는 변수에 대한 슬라이더 먼저 생성
    with col1:
      st.markdown("**내 고정지출 수준 분석**")
      input_total = 0
      for cate in inputs_with_values:
            default_value = user_inputs[input_fields[cate][1]]
            st.slider(cate, result.loc['min', input_fields[cate][1]], result.loc['max', input_fields[cate][1]], default_value)
            input_total += user_inputs[input_fields[cate][1]]
    # 입력 값이 없는 변수에 대한 슬라이더 생성
    with col2:
      st.markdown("**예상 변동지출**")
      min_total, max_total, mean_total = 0, 0, 0

      for cate in inputs_without_values:
            default_value = result.loc['mean', input_fields[cate][1]]
            st.slider(cate, result.loc['min', input_fields[cate][1]], result.loc['max', input_fields[cate][1]], default_value)
            min_total += result.loc['min', input_fields[cate][1]]
            max_total += result.loc['max', input_fields[cate][1]]
            mean_total += default_value
    with st.container():
      my_total_value = mean_total + input_total
      st.slider('**이번 달 예상 지출액**', (min_total+input_total), (max_total+input_total), (my_total_value))
  else:
    st.write("")



