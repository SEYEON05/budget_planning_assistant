import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pages import *
import time


if 'page' not in st.session_state: # 처음 페이지 들어가면 session_state가 비어있기 때문에 if not in문으로 키 'page'를 session_state에 할당해 시작페이지를 HOME으로 설정
  st.session_state['page'] = 'HOME'


@st.cache_data  # 데이터를 rerun할 때마다 다시 load할 필요없게 만듦.
def load_data():
  data = pd.read_csv("student_spending (1).csv", index_col=0)
  return data
data = load_data()

# 항목별 columns 보이기
# for key in data.columns:
#   if key == 'age':
#     'age:', set(data['age'])
#   elif key == 'gender':
#     'gender:', set(data['gender'])
#   elif key == 'year_in_school':
#     'year_in_school:', set(data['year_in_school'])
#   elif key == 'major':
#     'major:', set(data['major'])
#   elif key == 'preferred_payment_method':
#     'preferred_payment_method:', set(data['preferred_payment_method'])
# data

menus = {'HOME': home, '소비 패턴 분석': pattern_analyis, '내 지출 수준 분석': planner}
with st.sidebar:
  for menu in menus.keys():
    # 버튼을 클릭 시, menu에 클릭된 값이 할당됨. 
    if st.button(menu, use_container_width=True, type='primary' if st.session_state['page']==menu else 'secondary'):
      st.session_state['page'] = menu
      st.rerun()  # 자동 새로고침

for menu in menus.keys():
  if st.session_state['page']==menu:
    menus[menu]()