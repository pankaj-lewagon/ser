import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import joblib
from ser.data import extract_features
from ser.predict import return_predict, predict_proba,x_pred_preprocessing
import requests
import shutil

import streamlit as st

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def index():
    return {"greeting": "Hello world!!!"}

@app.post("/predict/")
async def predict(file: UploadFile  = File(...)):

    # Create generic 'ouput' + extension filename to avoid writing too many files on disk
    # As model can handle severeal audio file types we retrieve the extension form provided filename
    # pp=type(file.file)
    filename = 'output.' + str(file.filename)[-3:]
    print(filename)

    with open(filename,'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)



    # observed_emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

    x_pred_preprocessed = x_pred_preprocessing(filename)
    # prediction = return_predict(x_pred_preprocessed)
    # predicted_probas = predict_proba(observed_emotions, x_pred_preprocessed)
    # # return prediction, predicted_probas
    # ###delete pip.wav
    return{'hello world'}



if __name__ == "__main__":

    file_='/Users/pankajpatel/Documents/code/pankaj-lewagon/ser/raw_data/ravdess_data/Actor_01/03-01-01-01-01-01-01.wav'

    files={'file': open(file_,'rb') }
    url="http://localhost:8000/predict/"
    r=requests.post(url, files=files)
    print(f'requests output {r.text}')
