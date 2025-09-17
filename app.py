import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 

#loading from colab 
rf= joblib.load("sleep_quality_model.pkl")
feature_names = joblib.load("feature_names.pkl")
category_options = joblib.load("category_options.pkl")

#streamlit page setup
st.set_page_config(page_title = "Sleep Quality Predictor", layout = "centered")
st.title("Sleep Quality Predictor")
st.caption("Enter your infomration and I will estimate your sleep quality 0 to 10")

#debugger
#confirms everything is loaded properly, and the inputs math models expectations 
with st.sidebar.expander("Debug: show loaded info"):
    st.write("Expected feature columns:", feature_names)
    st.write("Category options:", {k: sorted(v) for k, v in category_options.items()})

st.header("Enter Your Info")

col1, col2 = st.columns(2) #splitting into 2 columns for better layout 

with col1: 
    age = st.number_input("Age", min_value=0, max_value=120, value = 30)
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value = 0.0, max_value = 24.0, value = 7.0, step=0.1)
    phys_activity = st.number_input("Physical Activity Level (minutes/day)", min_value = 0, max_value = 500, value = 60, step = 5)
    stress = st.number_input("Stress Level (1-10)", min_value = 1, max_value = 10, value = 5)

with col2:
    heart_rate = st.number_input("Heart rate (bpm)", min_value = 30, max_value = 220, value = 70, step =1)
    daily_steps = st.number_input("Daily Steps", min_value = 0, max_value = 50000, value = 8000, step = 500)
    systolic = st.number_input("Systolic BP", min_value = 70, max_value = 200, value = 120)
    diastolic = st.number_input("Diastolic BP", min_value = 40, max_value = 120, value = 80)

#now we add the categorical dropdowns
gender = st.selectbox("Gender", options = sorted(category_options["Gender"]))
occupation = st.selectbox("Occupation", options = sorted(category_options["Occupation"]))
bmi_cat = st.selectbox("BMI Category", options = sorted(category_options["BMI Category"]))
sleep_dis = st.selectbox("Sleep Disorder", options = sorted(category_options["Sleep Disorder"]))

#our model needs a df so we need to put everything into a dictionary first 
row = {} #empty dict
row["Age"]= age #takes whatever the user typed int age box and puts it in our dict under the key "Age"
row["Sleep Duration"]= sleep_duration 
row["Physical Activity Level"] = phys_activity
row["Stress Level"] = stress
row["Heart Rate"] = heart_rate 
row["Daily Steps"] = daily_steps
row["Systolic"] = systolic
row["Diastolic"] = diastolic

row["Gender_Male"] = 1 if gender =="Male" else 0 
row["Gemder_Female"] = 1 if gender =="Female" else 0

for occ in category_options["Occupation"]:
    row[f"Occupation_{occ}"] = 1 if occupation == occ else 0

for cat in category_options["BMI Category"]:
    row[f"BMI Category_{cat}"] = 1 if bmi_cat ==cat else 0 

for dis in category_options["Sleep Disorder"]:
    row[f"Sleep Disorder_{dis}"] = 1 if sleep_dis ==dis else 0


# turn users input dictionary into df 
#make sures it has the same columns in same order
user_df = pd.DataFrame([row])
user_df = user_df.reindex(columns=feature_names, fill_value=0)
#takes user inputs and rearramges to match exact column names and order that the model was trained


prediction = rf.predict(user_df)[0]
#rd:trained randomforest mo9del thats loaded from colab 

prediction = max(0.0, min(10.0, float(prediction)))
st.subheader("Predicted Sleep Quality")
st.success(f"Your predicted sleep quality is: {prediction:.2f} / 10")
#st.success: green box 


