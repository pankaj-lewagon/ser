import joblib
from ser.data import extract_features


#Load an audio file and transform it
def x_pred_preprocessing(audio_path):
    x_pred_preprocessed = extract_features(audio_path,
                                           mfcc=True,
                                           chroma=False,
                                           mel=True,
                                           temp=True)
    x_pred_preprocessed = x_pred_preprocessed.reshape(1, 552)
    return x_pred_preprocessed


#Function to predict
def return_predict(x_pred_preprocessed, model_path='MLP_model.joblib'):
    model = joblib.load(model_path)
    prediction = model.predict(x_pred_preprocessed)
    return prediction


if __name__ == '__main__':
    audio_path = 'OAF_back_angry.wav'
    x_pred_preprocessed = x_pred_preprocessing(audio_path)
    prediction = return_predict(x_pred_preprocessed)
    print(prediction)
