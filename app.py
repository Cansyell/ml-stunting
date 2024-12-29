import numpy as np
from flask import Flask, render_template, request
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__, template_folder='template')

# Memuat model dengan path yang benar
model = joblib.load(open("Model/stunting_klasifikasi.joblib", "rb"))

# Fungsi untuk mengonversi gender
def convert_gender(gender):
    if gender == "male":
        return 0  # 0 untuk male
    elif gender == "female":
        return 1  # 1 untuk female
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # Mengambil data input dari form
    age = int(request.form['Age(months)'])
    gender = convert_gender(request.form['Gender'])
    height = float(request.form['Height(cm)'])

    # Membuat fitur untuk prediksi
    feature = np.array([[age, gender, height]])

    # Prediksi menggunakan model
    prediction = model.predict(feature)

    # Mengambil prediksi pertama dari hasil yang dikembalikan
    prediction_result = prediction[0]

    # Tentukan rekomendasi berdasarkan hasil prediksi
    recommendation = ""
    if prediction == "tall":
        recommendation = "Tidak ada rekomendasi khusus, ketinggian sudah diatas rata-rata, sehat selalu."
    elif prediction == "normal":
        recommendation = "Tetap jaga tumbuh kembang anak anda dengan mengonsumsi makanan sehat seperti buah buahan lokal pride."
    elif prediction == "stunted":
        recommendation = "Perhatikan asupan gizi anak, pastikan mereka mendapatkan nutrisi yang cukup untuk tumbuh dengan baik."
    elif prediction == "severely stunted":
        recommendation = "Segera konsultasikan dengan dokter atau ahli gizi untuk mendapatkan penanganan lebih lanjut mengenai stunting yang dialami."

    # Mengirim hasil prediksi dan rekomendasi ke template
    return render_template("index.html", prediction_text="Your toddler is {}".format(prediction_result), recommendation_text=recommendation)

if __name__ == "__main__":
    app.run(debug=True)


