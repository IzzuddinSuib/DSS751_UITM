import streamlit as st

# Load EDA packages
import pandas as pd
import numpy as np
import os

# Load Data Viz packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

# Load Data
@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.subheader("Exploratory Data Analysis")
    # df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Telco_Churn_ DSS\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # df = load_data(r"C:\Users\Lenovo\Desktop\Telco_Churn_ DSS\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_encoded = load_data("WA_Fn-UseC_-Telco-Customer-Churn_clean.csv")
    # df_encoded = load_data(r"C:\Users\Lenovo\Desktop\Telco_Churn_ DSS\data\WA_Fn-UseC_-Telco-Customer-Churn_clean.csv")
    submenu = st.sidebar.selectbox("Submenu",['Descriptive','Plots'])
    if submenu == "Descriptive":
        st.dataframe(df)

        with st.expander("Data Types"):
            ###
            # # numerical
            # column_numerical = ['tenure', 'monthly_charges', 'total_charges']
            #
            # # categorical
            # column_categorical = list(df.columns)
            # column_categorical.remove('tenure')
            # column_categorical.remove('MonthlyCharges')
            # column_categorical.remove('TotalCharges')
            #
            # data_type_general = dict()
            #
            # for col in df.columns:
            #     if col in column_numerical:
            #         data_type_general[col] = 'numerical'
            #     else:
            #         data_type_general[col] = 'categorical'
            #
            # tmp = pd.Series(data_type_general)
            # data_type_general = pd.DataFrame(tmp).T.rename({0: 'general data types'})
            # data_type_general
            # st.dataframe(data_type_general)

            ###
            st.dataframe(df.dtypes.astype(str))

        with st.expander("Desciptive Summary"):
            st.dataframe(df_encoded.describe())

        with st.expander("Duplicate Values"):
            st.code(df.duplicated().sum())

        with st.expander("Total Data"):
            st.dataframe(df.count().T.rename({0:'total data'}))

        with st.expander("Total Null Values"):
            st.dataframe(df.isna().sum().T.rename({0:'total null'}))

        with st.expander("Null Values Percentage"):
            st.dataframe((100*df.isna().sum()/df.shape[0]).T.rename({0:'percentage null'}))

        with st.expander("Data Range For Numerical Data"):
            df.loc[df['MonthlyCharges'].isnull(), 'TotalCharges'] = 0
            column_numerical = {'tenure','MonthlyCharges','TotalCharges'}
            variation_numerical = dict()

            for col in column_numerical:
                tmp = f'{df[col].min()} - {df[col].max()}'
                variation_numerical[col] = tmp

            tmp = pd.Series(variation_numerical)
            data_variation_numerical = pd.DataFrame(tmp).T.rename({0: 'data variation'})
            data_variation_numerical
            st.dataframe(data_variation_numerical)

        with st.expander("Overall Data Variation"):
            column_categorical = {'gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod', 'Churn' }
            variation_categorical = dict()

            for col in column_categorical:
                tmp = df[col].unique().tolist()
                tmp.sort()
                variation_categorical[col] = ', '.join(str(item) for item in tmp)

            tmp = pd.Series(variation_categorical)
            data_variation_categorical = pd.DataFrame(tmp).T.rename({0: 'data variation'})
            data_variation_categorical

            data_variation = pd.concat([data_variation_numerical, data_variation_categorical], axis=1)
            data_variation
            final_var = pd.concat([data_variation_numerical.rename({'data variation': 'range'}),
                       data_variation_categorical.rename({'data variation': 'variation'})], axis=0).fillna('-').reindex(
                df.columns, axis=1).T
            st.dataframe(final_var)

        with st.expander("Dataset Summary"):
            st.dataframe(df['gender'].value_counts())

        with st.expander("Gender Distribution"):
            st.dataframe(df['gender'].value_counts())

        with st.expander("Tenure Distribution"):
            st.dataframe(df['tenure'].value_counts())

    elif submenu == "Plots":
        st.subheader("Plots")

        # Layouts
        col1,col2 = st.columns([2,1])

        with col1:
            with st.expander("Dist Plot of Gender"):
                # st.write("Tell me why")
                # Using Seaborn
                fig = plt.figure()
                sns.countplot(df['gender'])
                st.pyplot(fig)

                gen_df = df['gender'].value_counts()
                gen_df = gen_df.reset_index()
                gen_df.columns = ["Gender Type", "Counts"]
                # st.dataframe(gen_df)

                p1 = px.pie(gen_df,names='Gender Type', values='Counts')
                st.plotly_chart(p1, use_container_width=True)

            # For Tenure Distribution
            with st.expander("Dist  Plot of  Tenure"):
                fig = plt.figure()
                sns.countplot(df['tenure'])
                st.pyplot(fig)

        with col2:
            with st.expander("Gender Distribution"):
                st.dataframe(gen_df)

            with st.expander("Tenure Distribution"):
                st.dataframe(df['tenure'].value_counts())

        # Freq Dist
        with st.expander("Frequency Dist of Age"):
            tenure_df = df['tenure'].value_counts().reset_index().rename(columns={'index':'tenure_period', 'tenure':'count'})
            st.dataframe(tenure_df)
            p2 = px.bar(tenure_df,x='tenure_period', y='count')
            st.plotly_chart(p2)

        # Outlier Detection
        with st.expander("Outlier Detection (Monthly Charges)"):
            p3 = px.box(df, x='MonthlyCharges')
            st.plotly_chart(p3)

        with st.expander("Outlier Detection (Total Charges)"):
            p4 = px.box(df, x='TotalCharges', color='gender')
            st.plotly_chart(p4)

        # Correlation
        with st.expander("Correlation Plot"):
            corr_matrix = df.corr()
            fig = plt.figure(figsize=(20,10))
            sns.heatmap(corr_matrix, annot=True)
            st.pyplot(fig)

            p5 = px.imshow(corr_matrix)
            st.plotly_chart(p5)

        #delete later
        with st.expander("check df"):
            st.dataframe(df.head())