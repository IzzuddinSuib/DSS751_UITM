# streamlit run main.py

# Core packages
import streamlit as st
import streamlit.components.v1 as stc
import os

# Import mini apps
from eda_app import run_eda_app
from ml_app import run_ml_app
from PIL import Image

# HTML template
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Customers Churn Data App </h1>
		<h4 style="color:white;text-align:center;">Churn Analysis & Prediction </h4>
		</div>
		"""



def main():
    stc.html(html_temp)

    menu = ["Home","EDA","ML","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("""
        			### Telco Customer Churn Predictor App
        			This dataset contains the customer data from a made-up telco company. This company provides various services such as streaming, phone, and internet services.
        			#### Datasource
        				- https://www.kaggle.com/datasets/blastchar/telco-customer-churn
        			#### App Content
        				- EDA Section: Exploratory Data Analysis of Data
        				- ML Section: ML Predictor App

        			""")
    elif choice == "EDA":
        run_eda_app()
    elif choice == "ML":
        run_ml_app()
    else:
        st.subheader("About")
        image = Image.open('pbout.PNG')
        # image= r'C:\Users\Lenovo\Desktop\Telco_Churn_ DSS\photos\about.png'
        st.image(image, use_column_width=True)

if __name__=='__main__':
    main()