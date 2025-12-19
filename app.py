from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'breast_cancer_prediction_secret_key'

# Load model dan scaler
model = joblib.load("model_gnb.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil semua input fitur dari form
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Prediksi menggunakan GNB
    prediction_class = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    
    # Dapatkan kelas-kelas yang digunakan model
    classes = model.classes_
    
    # Cari indeks kelas yang diprediksi
    class_idx = np.where(classes == prediction_class)[0][0]
    
    # Ambil probabilitas untuk kelas yang diprediksi
    probability = float(prediction_proba[class_idx])
    
    # Tentukan prediksi dan status berdasarkan kelas
    # GNB biasanya mengembalikan 0 untuk Benign dan 1 untuk Malignant
    # atau 'B' untuk Benign dan 'M' untuk Malignant
    if prediction_class == 1 or prediction_class == 'M':
        prediction = 'Malignant (Ganas)'
        status = 'danger'
        recommendation = 'Segera konsultasikan dengan dokter spesialis onkologi untuk pemeriksaan lebih lanjut dan penanganan yang tepat.'
    else:
        prediction = 'Benign (Jinak)'
        status = 'success'
        recommendation = 'Tetap lakukan pemeriksaan rutin sesuai anjuran dokter untuk memantau kondisi kesehatan Anda.'
    
    # Simpan hasil ke session untuk ditampilkan di result.html
    session['prediction'] = prediction
    session['probability'] = probability
    session['status'] = status
    session['recommendation'] = recommendation
    
    return redirect(url_for('result'))

@app.route('/result')
def result():
    prediction = session.get('prediction', 'Tidak ada hasil')
    probability = session.get('probability', 0)
    status = session.get('status', 'info')
    recommendation = session.get('recommendation', '')
    
    return render_template('result.html', 
                         prediction=prediction, 
                         probability=probability,
                         status=status,
                         recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)