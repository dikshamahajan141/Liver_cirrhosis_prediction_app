import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.tracebacklimit = 0

st.set_page_config(
    page_title="Multi-predict app", 
    page_icon="house",
)
data1=pd.read_csv('Liver cirrhosis UCI Dataset.csv')
data=data1.head()
st.title("WELCOME!")
st.header("PREDICT YOUR HEALTH STATE")
st.subheader("~We care for you!")
st.image("img11.png",width=500)

if st.button("LEARN ABOUT LIVER CIRRHOSIS"):
     st.write("Cirrhosis is scarring (fibrosis) of the liver caused by long-term liver damage. The scar tissue prevents the liver working properly. Cirrhosis is sometimes called end-stage liver disease because it happens after other stages of damage from conditions that affect the liver, such as hepatitis.") 


#st.markdown("WELCOME!You can now predict the diseases related to the important organs of your body using our app. ")
st.write("Have a look at the data sample we use to cater you the results:")
if st.button("Sample Records"):
     st.table(data)
st.image("Liver-Cirrhosis.jpg.webp",width=650)
st.header("About us:")
st.markdown(':green[This website uses a machine learning model well trained by our team to help machine learn from the recorded data of patients and predict your state. Kindly note that the results shown are not 100% true and if the disease is predicted, make sure to take consultation from a doctor. Stay healthy and safe.]')