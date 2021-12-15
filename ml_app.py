import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_ml_app():
    st.subheader('머신 러닝 예측')

    # 데이터 입력
    # 입력-성별
    gender_data = ['여성', '남성']
    gender = st.radio('성별', gender_data)
    if gender == gender_data[0]:
        gen_set = 0
        gen_show = gender_data[0]
    elif gender == gender_data[1]:
        gen_set = 1
        gen_show = gender_data[1]

    # 입력-나이
    age_set = st.number_input('나이 입력', 18, 80)
    sal_set = st.number_input('연봉 입력', step=100, help='너무 낮거나 높은 값을 입력하면 정확한 예측이 되지 않을 수 있습니다.')
    debt_set = st.number_input('카드 부채', 0, step=100)
    net_set = st.number_input('자산 입력', step=100, help='너무 낮거나 높은 값을 입력하면 정확한 예측이 되지 않을 수 있습니다.')

    # 모델을 통해 예측
    new_data = np.array([gen_set, age_set, sal_set, debt_set, net_set])
    new_data = new_data.reshape(1, new_data.size)

    scaler_X = joblib.load('data/scaler_X.pkl')
    scaler_y = joblib.load('data/scaler_y.pkl')
    regressor = joblib.load('data/regressor.pkl')

    new_data = scaler_X.transform(new_data)

    y_pred = regressor.predict(new_data)
    print(y_pred)

    y_pred = scaler_y.inverse_transform(y_pred.reshape(1,1))

    # 예측 결과 출력
    st.write(f'{str(age_set)}세 {gen_show}.')
    st.write(f'연봉 : {str(sal_set)} $')
    st.write(f'부채 : {str(debt_set)} $')
    st.write(f'자산 : {str(net_set)} $')
    st.write(f'위의 인물이 구매할법한 차량의 가격은...?')
    
    result = y_pred[0,0].round(1)
    if st.button('예측'):
        st.write(f'{result}$ 정도의 자동차면 적당합니다.')

    