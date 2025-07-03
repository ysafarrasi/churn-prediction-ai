# Skrip Prediksi Churn Pelanggan - Versi Rapih dan Modular

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://drive.google.com/uc?export=download&id=1-hU2LbBBbip_9bdxko6Z6RrRrsBt90IC"
df = pd.read_csv(url)
x = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender', 'Exited'], axis=1)
y = df['Exited']

def evaluate_model(model, x, y, n_splits=5, scale=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, precisions, recalls = [], [], []
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x) if scale else x.values
    for train_idx, test_idx in skf.split(x_scaled, y):
        x_train, x_test = x_scaled[train_idx], x_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro'))
        recalls.append(recall_score(y_test, y_pred, average='macro'))
    print(f"{model.__class__.__name__} - Accuracy: {np.mean(accuracies):.2f}, Precision: {np.mean(precisions):.2f}, Recall: {np.mean(recalls):.2f}")

models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel="rbf"),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
]

for model in models:
    evaluate_model(model, x, y)

print("\n=== Skrip selesai dieksekusi ===")
