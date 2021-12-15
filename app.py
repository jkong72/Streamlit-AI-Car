import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from eda_app import run_eda_app


def main():
    st.title('자동차 가격 예측')
    
    # 사이드바
    menu = ['Home', 'EDA', 'ML']
    menu_choice = st.sidebar.selectbox('메뉴', menu)
    
    # menu Home
    if menu_choice == menu[0]:
        st.write ('고객 정보와 차량 구입 데이터를 바탕으로 고객의 정보를 입력하면 어느정도의 차량을 구매할지 예측하는 인공지능 서비스입니다.')
        st.write ('왼쪽 메뉴를 선택해 작업을 시작할 수 있습니다.')

    # menu EDA
    if menu_choice == menu[1]:
        run_eda_app()    
                    
    # mene ML
    if menu_choice == menu[2]:
        pass


if __name__ == '__main__':
    main()