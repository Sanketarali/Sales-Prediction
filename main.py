import streamlit as st
import joblib

model = joblib.load('model_joblib_test')

st.title("SalesPrediction")

tv = st.number_input("TV")
radio= st.number_input("Radio")
newspaper= st.number_input("Newspaper")



if st.button("Predict"):
   
   
    result = model.predict([[tv,radio,newspaper]])
    st.write("Sales:" , result[0])
