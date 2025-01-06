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
        recommendation = "There are no special recommendations, height is above average, always healthy."
    elif prediction == "normal":
        recommendation = "Keep your child's growth and development by consuming healthy foods such as local fruits."
    elif prediction == "stunted":
        recommendation = "Pay attention to your child's nutritional intake, make sure they get enough nutrition to grow well."
    elif prediction == "severely stunted":
        recommendation = "Immediately consult a doctor or nutritionist to get further treatment regarding the stunting you are experiencing."

    # Mengirim hasil prediksi dan rekomendasi ke template
    return render_template("index.html", 
                           prediction_text="Based on the data you entered (Age: {} Month, Gender: {}, Height: {} cm), your toddler is classified as {}".format(age, "Male" if gender == 0 else "Female", height, prediction_result),
                           recommendation_text=recommendation)

if __name__ == "__main__":
    app.run(debug=True)


