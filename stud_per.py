import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

"""
mongoDB 
    username - mahipalsinhthakor, 
    password - k98sXehgLcxFVg4l

    mongodb+srv://atlas-sample-dataset-load-67fbb5c9a939f22be8f33a91:<db_password>@datascienceeuron.fcffh8n.mongodb.net/?retryWrites=true&w=majority&appName=dataScienceEuron

    uri = "mongodb+srv://<db_username>:<db_password>@datascienceeuron.fcffh8n.mongodb.net/?appName=dataScienceEuron"

"""

# uri = "mongodb+srv://sudhanshu:sudh1234@cluster0.pnq4z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# client = MongoClient(uri, server_api=ServerApi('1'))
# db = client['student']
# collection = db["student_pred"]

uri = "mongodb+srv://mahipal:mahipal@datascienceeuron.fcffh8n.mongodb.net/?retryWrites=true&w=majority&appName=dataScienceEuron"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'),
    tls=True,
    tlsAllowInvalidCertificates=True)

db = client['student']
collection = db["student_pred"]

def load_model():
    with  open("student_lr_final_model.pkl",'rb') as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocesssing_input_data(data, scaler, le):
    data['Extracurricular Activities']= le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocesssing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("student performnce perdiction")
    st.write("enter your data to get a prediction for your performance")
    
    hour_sutdied = st.number_input("Hours studied",min_value = 1, max_value = 10 , value = 5)
    prvious_score = st.number_input("previous score",min_value = 40, max_value = 100 , value = 70)
    extra = st.selectbox("extra curri activity" , ['Yes',"No"])
    sleeping_hour = st.number_input("sleeping hours",min_value = 4, max_value = 10 , value = 7)
    number_of_peper_solved = st.number_input("number of question paper solved",min_value = 0, max_value = 10 , value = 5)
    
    if st.button("predict-your_score"):
        user_data = {
            "Hours Studied":hour_sutdied,
            "Previous Scores":prvious_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleeping_hour,
            "Sample Question Papers Practiced":number_of_peper_solved
        }
        prediction = predict_data(user_data)

        user_data['prediction'] = prediction
        collection.insert_one(user_data)

        st.success(f"your prediciotn result is {prediction}")
    
if __name__ == "__main__":
    main()
    

