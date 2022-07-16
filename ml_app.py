# Import core packages
import streamlit as st

# Load ML packages
import joblib
import os
# import sklearn

# Load EDA packages
import numpy as np

attrib_info = """
"""

def get_fvalue(val):
    feature_dict = {"No":0, "Yes":1}
    for key,value in feature_dict.items():
        if val ==key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

# Load ML Models
@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

label_dict = {"No":0,"Yes":1}
gender_map = {"Female":0,"Male":1}
MultipleLines_map = {"No":0,"Yes":1,"No phone service":2}
InternetService_map = {"No":0,"DSL":1,"Fiber optic":2}
Others_map = {"No":0,"Yes":1,"No internet service":2}
target_label_map = {"No":0,"Yes":1}

def run_ml_app():
    st.subheader("ML Prediction")
    # st.write("It's successful")
    # st.success("Wow it is a cool apps")

    with st.expander("Attribute Info"):
        st.markdown(attrib_info)

    # Layout
    col1,col2 = st.columns(2)

    with col1:
        # age = st.number_input("Age",10,100)
        gender = st.radio("Gender",["Female","Male"])
        partner = st.radio("Partner",["No","Yes"])
        SeniorCitizen =  st.radio("Senior_Citizen",["No","Yes"])
        dependents = st.radio("Dependents", ["No", "Yes"])
        PhoneService = st.radio("Phone_Service", ["No", "Yes"])
        MultipleLines = st.selectbox("Multiple_Lines", ["No", "Yes","No phone service"])
        InternetService = st.selectbox("Internet_Service", ["No", "DSL", "Fiber optic"])

    with col2:
        OnlineSecurity = st.selectbox("Online_Security", ["No", "Yes","No internet service"])
        OnlineBackup = st.selectbox("Online_Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device_Protection", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech_Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming_TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming_Movies", ["No", "Yes", "No internet service"])
        PaperlessBilling = st.radio("Paperless_Billing", ["No", "Yes"])

    with st.expander("Your Selected Options"):
        result = {
                  # 'Age':age,
                  'gender':gender,
                  'partner':partner,
                  'SeniorCitizen':SeniorCitizen,
                  'dependents':dependents,
                  'PhoneService':PhoneService,
                  'MultipleLines':MultipleLines,
                  'InternetService':InternetService,
                  'OnlineSecurity':OnlineSecurity,
                  'OnlineBackup': OnlineBackup,
                  'DeviceProtection': DeviceProtection,
                  'TechSupport': TechSupport,
                  'StreamingTV':StreamingTV,
                  'StreamingMovies': StreamingMovies,
                  'PaperlessBilling': PaperlessBilling
                  }

        st.write(result)

        encoded_result = []
        for i in result.values():
            if type(i) == int:
                encoded_result.append(i)
            elif i in ["Female","Male"]:
                res = get_value(i,gender_map)
                encoded_result.append(res)
            elif i in ["No","Yes","No phone service"]:
                res = get_value(i,MultipleLines_map)
                encoded_result.append(res)
            elif i in ["No","DSL","Fiber optic"]:
                res = get_value(i,InternetService_map)
                encoded_result.append(res)
            elif i in ["No","Yes","No internet service"]:
                res = get_value(i,Others_map)
                encoded_result.append(res)
            else:
                encoded_result.append(get_fvalue(i))

        st.write(encoded_result)

    with st.expander("Prediction Result"):
        single_sample = np.array(encoded_result).reshape(1,-1)
        st.write(single_sample)

        model = load_model("logistic_regression_model_telco_15_june_2022.pkl")
        prediction = model.predict(single_sample)
        pred_prob = model.predict_proba(single_sample)
        st.write(prediction)
        st.write(pred_prob)

        if prediction == 1:
            st.success("Will Probably Not Churn".format(prediction[0]))
            pred_probability_score = {"Churn Risk":pred_prob[0][0]*100,
                                      "Not Churn Risk":pred_prob[0][1]*100}
            st.write(pred_probability_score)
        else:
            st.warning("Will Probably Churn".format(prediction[0]))
            pred_probability_score = {"Churn Risk": pred_prob[0][0]*100,
                                      "Not Churn Risk": pred_prob[0][1]*100}
            st.write(pred_probability_score)

