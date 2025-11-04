"""Generate a synthetic dataset for patient readmission prediction."""
import numpy as np
import pandas as pd
import os
rng = np.random.RandomState(42)
n = 2000
patient_id = np.arange(1, n+1)
age = rng.randint(18, 95, size=n)
sex = rng.choice(['M','F'], size=n)
num_prev_adm = rng.poisson(1.2, size=n)
length_of_stay = rng.randint(1, 15, size=n)
comorbidity_score = rng.poisson(2.0, size=n)
med_count = rng.randint(0, 12, size=n)
discharge_disposition = rng.choice(['home','home_with_care','nursing_home','other'], size=n, p=[0.7,0.15,0.1,0.05])
avg_lab_result = rng.normal(loc=100, scale=15, size=n)
risk = (0.02*(age-50) + 0.3*(num_prev_adm) + 0.05*(comorbidity_score) + 0.04*(med_count) + 0.1*(length_of_stay>7).astype(int))
prob = 1 / (1 + np.exp(- ( -3 + risk )))
readmitted_30d = (rng.rand(n) < prob).astype(int)
df = pd.DataFrame({
    'patient_id': patient_id,
    'age': age,
    'sex': sex,
    'num_prev_adm': num_prev_adm,
    'length_of_stay': length_of_stay,
    'comorbidity_score': comorbidity_score,
    'med_count': med_count,
    'discharge_disposition': discharge_disposition,
    'avg_lab_result': avg_lab_result,
    'readmitted_30d': readmitted_30d
})
os.makedirs('data', exist_ok=True)
df.to_csv('data/raw.csv', index=False)
print('Generated data/raw.csv with', n, 'rows')
