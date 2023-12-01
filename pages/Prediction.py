import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import sys
sys.tracebacklimit = 0
import pickle

#loaded_model = joblib.load("trained_model.joblib")
loaded_model = pickle.load(open('model.sav','rb'))

df= pd.read_csv("Liver cirrhosis UCI Dataset.csv")

newdf=df.dropna(axis=0, how="any")

X = newdf.drop(['Selector field used to split the data into two sets (labeled by the experts)'],axis=1)
classification = newdf['Selector field used to split the data into two sets (labeled by the experts)']

categorical_cols = ['GENDER']  # Replace with the names of your categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(classification)

# Scale the features using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert the encoded classification column to probability values of "yes" in percentage
#class_probabilities = classification/ (classification.max() - classification.min())
class_probabilities = y_encoded / (y_encoded.max() - y_encoded.min())
# Create a new DataFrame with scaled features and probability values
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled['Probability_of_Yes'] = class_probabilities
y=df_scaled['Probability_of_Yes']
X = df_scaled.drop(['Probability_of_Yes'], axis=1)

poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)

import streamlit as st
st.set_page_config(
        page_title="Liver cirrhosis", 
    )
st.title("LIVER CIRRHOSIS")
st.image("liverimg.png",width=400)

st.write("Kindly fill below  to get results:")

age=st.number_input("Age",min_value=1)
gen = int(st.number_input("Enter gender (0 for female, 1 for male): "))
TB=float(st.number_input("Total bilrubin"))
DB=float(st.number_input("Direct bilrubin"))
AAP=float(st.number_input("Alkphos Alkaline Phosphotase",min_value = 0.0))
Sgpt=float(st.number_input("Sgpt Alamine Aminotransferase",min_value = 0.0))
sgot=float(st.number_input("Sgot Aspartate Aminotransferase",min_value = 0.0))
tp=float(st.number_input("Total Protiens",min_value = 0.0))
alb=float(st.number_input("Albumin",min_value = 0.0))
ratio=float(st.number_input("A/G Ratio",min_value = 0.0))
cb=st.checkbox("I agree to provide my test results")     
user_input=np.array([[age,gen,TB,DB,AAP,Sgpt,sgot,tp,alb,ratio]])

user_input_poly = poly.transform(user_input)
user_input_scaled = scaler.transform(user_input_poly)

# Use the trained stacking model to make predictions
probability = loaded_model.predict_proba(user_input_scaled)[0][1]
if cb & st.button("Submit and Predict"):
    st.success(f"Predicted probability: {probability}")
    if probability < 0.1204:
        st.markdown(':green[No need to worry as of now as the disease is not predicted but make sure to have a healthy diet to stay healthy :)]')
    else:
        st.markdown(':red[The model predicts that you have a fair chance of having the disease. Make sure to consult a doctor to confirm results and get a treatment as soon as possible.]')
