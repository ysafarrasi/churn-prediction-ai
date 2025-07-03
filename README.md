# Churn Prediction with Machine Learning

Prediksi churn pelanggan menggunakan beberapa algoritma Machine Learning sebagai bagian dari studi kasus Data Science.

## 📌 Deskripsi Proyek
Proyek ini bertujuan mengidentifikasi pelanggan yang berpotensi keluar (churn) dari sebuah institusi keuangan menggunakan dataset *Churn_Modelling*. Berbagai algoritma dikembangkan dan dibandingkan untuk mendapatkan model terbaik.

## 🔍 Model yang Digunakan
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Neural Network (MLP)
- Naive Bayes Classifier

## 📊 Dataset
Dataset publik yang digunakan dapat diunduh melalui:
[Download Dataset - Google Drive](https://drive.google.com/uc?export=download&id=1-hU2LbBBbip_9bdxko6Z6RrRrsBt90IC)

Fitur penting mencakup informasi demografis, saldo, dan aktivitas pelanggan. Target variabel adalah `Exited` (1 = churn, 0 = tetap).

## 🛠️ Teknologi & Library
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## 🚀 Cara Menjalankan
1. Install dependensi:
   ```
   pip install -r requirements.txt
   ```
2. Jalankan skrip:
   ```
   python churn_prediction.py
   ```

Model akan dilatih dan dievaluasi menggunakan:
✅ Stratified K-Fold Cross Validation  
✅ Metode evaluasi: Akurasi, Presisi, Recall  
✅ Visualisasi Confusion Matrix  

## 🎯 Tujuan
- Membangun model prediksi churn yang akurat.
- Membandingkan kinerja berbagai algoritma klasifikasi.
- Menjadi bagian dari portofolio AI & Data Science.
