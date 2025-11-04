"""Simple FastAPI server to serve predictions. POST JSON with patient features and get risk probability back."""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

app = FastAPI()

class Patient(BaseModel):
    age: float
    sex: str
    num_prev_adm: int
    length_of_stay: int
    comorbidity_score: int
    med_count: int
    discharge_disposition: str
    avg_lab_result: float

model = None
try:
    model = joblib.load('models/model.joblib')
except Exception as e:
    print('Model not loaded:', e)

@app.post('/predict')
def predict(p: Patient):
    df = pd.DataFrame([p.dict()])
    df = pd.get_dummies(df, columns=['discharge_disposition'], drop_first=True)
    try:
        prob = model.predict(df)[0]
        return {'risk_probability': float(prob)}
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
