#using stramlit library to build app
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder


def load_model():
    with open("student_lr_model.pkl",'rb') as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title('Student Performance Prediction')
    st.write("Enter your data to get a prediction for your performance")

    hour_studied = st.number_input('Hours studied',min_value = 1, max_value = 10, value = 5)
    previous_score = st.number_input('Pevious Score',min_value = 40, max_value = 100, value = 70)
    extra = st.selectbox("Extra Curricular Activity" , ['Yes','No'])
    sleeping_hour = st.number_input('Sleeping Hours',min_value = 4, max_value = 10, value = 7)
    number_of_paper_solved = st.number_input('Number of Question Paper Solved',min_value = 0, max_value = 10, value = 5)

    if st.button("Predict-Your-Score"):
        user_data = {
            "Hours Studied":hour_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleeping_hour,
            "Sample Question Papers Practiced":number_of_paper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"Your Prediction Result is {prediction}")

if __name__ == "__main__":
    main()