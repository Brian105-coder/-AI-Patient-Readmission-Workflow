"""Evaluation script: loads model and test data, outputs confusion matrix and plots ROC curve"""
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def main(model_path='models/model.joblib', test_csv='data/processed.csv'):
    df = pd.read_csv(test_csv)
    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])
    X = df.drop(columns=['readmitted_30d'])
    y = df['readmitted_30d']
    model = joblib.load(model_path)
    probs = model.predict(X)
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(y, preds)
    print('Confusion Matrix:\n', cm)
    print('\nClassification Report:\n', classification_report(y, preds))
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    print('ROC AUC:', roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('docs/roc_curve.png')
    print('Saved ROC curve to docs/roc_curve.png')

if __name__ == '__main__':
    main()
