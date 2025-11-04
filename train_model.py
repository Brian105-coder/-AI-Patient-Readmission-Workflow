"""Train a LightGBM model on data/processed.csv and save model artifact and prints metrics."""
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main(infile='data/processed.csv', model_out='models/model.joblib'):
    df = pd.read_csv(infile)
    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])
    X = df.drop(columns=['readmitted_30d'])
    y = df['readmitted_30d']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lgb_train = lgb.Dataset(X_train, y_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    model = lgb.train(params, lgb_train, num_boost_round=200)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    probs = model.predict(X_test)
    preds = (probs > 0.5).astype(int)
    print('Accuracy:', accuracy_score(y_test, preds))
    print('Precision:', precision_score(y_test, preds))
    print('Recall:', recall_score(y_test, preds))
    print('F1:', f1_score(y_test, preds))
    print('ROC AUC:', roc_auc_score(y_test, probs))

if __name__ == '__main__':
    main()
