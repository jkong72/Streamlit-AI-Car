import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda_app():
    # 레이아웃
    st.title('EDA')
    st.subheader('선택적 자료 분석')
    df = pd.read_csv('data/Car_Purchasing_Data.csv')

    # 사이드바 라디오 옵션
    radio_menu = ['데이터프레임', '통계치']
    EDA_radio_type = st.sidebar.radio('데이터 유형', radio_menu)


    if EDA_radio_type == radio_menu[0]:
        # 사이드바 라디오 데이터프레임
        # 항목 레이아웃
        optioncol1, optioncol2 = st.columns(2)
        with optioncol1:
            selected_col = st.multiselect('항목', df.columns)

        with optioncol2:
            sex_list = ['female', 'male', 'both']
            selected_sex = st.selectbox('성별', sex_list, index=2)

        # 숫자 범위 지정
        df_numeric = df.iloc[:,4:]
        df_numeric = df_numeric.columns
        selected_num = st.selectbox('수치 지정', df_numeric)
        rangecol1, rangecol2 = st.columns(2)
        with rangecol1:
            num_to = st.number_input ('이상', value=df[selected_num].min(), step=1.0)
            if num_to < 0:
                num_to = 0
            if num_to == 0:
                num_to = df[selected_num].min()

        with rangecol2:
            num_from = st.number_input ('이하', value=df[selected_num].max(), step=1.0)
            if num_from == 0:
                num_from = df[selected_num].max()

        #버튼
        btncol1, btncol2, btncol3, btncol4 = st.columns(4)
        with btncol1:
            if st.button('최대로 맞추기'):
                num_to = df[selected_num].max()
        with btncol2:
            if st.button('최소값 초기화'):
                if num_to < 0:
                    num_to = 0
                if num_to == 0:
                    num_to = df[selected_num].min()
        with btncol3:
            if st.button('최소로 맞추기'):
                num_from = df[selected_num].min()
        with btncol4:
            if st.button('최대값 초기화'):
                num_from = df[selected_num].max()

                   
        # df 가공(성별)
        if selected_sex == 'female':
            df = df.loc[df['Gender']==0, ]
        elif selected_sex == 'male':
            df = df.loc[df['Gender']==1, ]
        else:
            pass

        # df 가공(컬럼)
        if len(selected_col) !=0:
            df = (df[selected_col])


        # df 가공(수치)
        if selected_num in df.columns:
            df = df.loc[df[selected_num]>=num_to]
            df = df.loc[df[selected_num]<=num_from]


        st.subheader('찾은 자료')
        if len(selected_col) != 0:
            st.write(', '.join(selected_col))
        else:
            st.write('모두')

        if selected_sex == selected_sex[0]:
            sexinf = '여성'
        elif selected_sex == selected_sex[1]:
            sexinf = '남성'
        else:
            sexinf = '사람'
        st.write(f'{selected_num}가 {num_to} 이상 {num_from} 이하인 {sexinf}')

        st.dataframe(df)

        isnum = df.dtypes != object
        isnum = df.columns[isnum]
        df_corr = df[isnum]
        if 'Gender' in selected_col:
            df_corr = df_corr.drop('Gender', axis=1)

        st.dataframe(df_corr.corr())
        
        # 차트 표시
        fig1 = sns.pairplot(data = df_corr)
        st.pyplot(fig1)
    
    # 사이드바 라디오 통계치
    if EDA_radio_type == radio_menu[1]:
        isnum = df.dtypes != object
        isnum = df.columns[isnum]
        df_corr = df[isnum]
        st.dataframe(df.describe())


    # st.subheader('인물 검색')
    # word = st.text_input('검색어를 입력하세요')
    # # way 1
    # word = word.lower()
    # df_searched = df.loc[df['Customer Name']==df['Customer Name'].str.lower.str.contains(word),:]
    # st.datafream(df_searched)



