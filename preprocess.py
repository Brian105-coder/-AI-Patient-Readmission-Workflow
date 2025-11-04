"""Preprocessing script: reads data/raw.csv and writes data/processed.csv"""
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

def main(infile='data/raw.csv', outfile='data/processed.csv', artifacts_dir='models'):
    os.makedirs(artifacts_dir, exist_ok=True)
    df = pd.read_csv(infile)
    df = df.copy()
    ids = df['patient_id']
    df = df.drop(columns=['patient_id'])
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    num_cols.remove('readmitted_30d')
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    joblib.dump(imputer, os.path.join(artifacts_dir, 'imputer.joblib'))
    df = pd.get_dummies(df, columns=['discharge_disposition'], drop_first=True)
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.joblib'))
    df['patient_id'] = ids
    cols = ['patient_id'] + [c for c in df.columns if c!='patient_id']
    df = df[cols]
    df.to_csv(outfile, index=False)
    print('Wrote', outfile)

if __name__ == '__main__':
    main()
