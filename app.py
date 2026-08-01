from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
# SINKRONISASI BERSAMA REACT: CORS diaktifkan agar API bisa ditembak oleh pnpm/Vite React kamu
CORS(app)  

print("🔄 Menyiapkan persenjataan model h5 Bisindo...")

# Muat kedua model secara global sewaktu server start up
# Pastikan nama file model .h5 kamu di folder sama persis seperti ini ya beb!
models = {
    'kata': tf.keras.models.load_model("model_bisindo_kata.h5"),
    'alphabet': tf.keras.models.load_model("model_bisindo_alphabet.h5")
}

# Definisikan array label manual jika file .npy kamu belum siap
# Kamu bisa ganti isi list di bawah ini dengan urutan label kelas alphabet dan kata aslimu!
classes = {
    'alphabet': ['A', 'B', 'C', 'D', 'E'],  # <--- Sesuaikan dengan label alphabet kamu
    'kata': ['Halo', 'Terima Kasih', 'Sama-sama', 'Saya', 'Belajar']  # <--- Sesuaikan dengan label kata kamu
}

print("✅ Semua model sukses masuk memori API Flask!")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    landmarks = data.get('landmarks')
    mode = data.get('mode', 'alphabet') # Default ke alphabet kalau tidak terdefinisi dari React
    
    # Validasi input 63 fitur koordinat (21 titik hand landmarks x 3 dimensi x/y/z)
    if not landmarks or len(landmarks) != 63:
        return jsonify({'error': 'Data koordinat tidak valid atau kurang dari 63 fitur'}), 400
        
    if mode not in models:
        return jsonify({'error': 'Mode model tidak dikenali'}), 400

    # Format input data ke format numpy array sesuai kebutuhan arsitektur LSTM
    input_data = np.array([landmarks])
    input_data = np.nan_to_num(input_data)
    
    # Reshape data agar memiliki dimensi waktu (batch_size, timesteps, features) -> (1, 1, 63)
    input_data_reshaped = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])
    
    # Eksekusi prediksi berdasarkan otak model h5 yang dipilih React
    model = models[mode]
    label_kunci = classes[mode]
    
    prediction = model.predict(input_data_reshaped, verbose=0)
    class_index = np.argmax(prediction)
    
    # Ambil teks label hasil prediksi
    if class_index < len(label_kunci):
        prediksi_teks = label_kunci[class_index]
    else:
        prediksi_teks = f"Kelas_{class_index}" # Fallback jika index melebihi kelas dummy
        
    confidence = float(prediction[0][np.argmax(prediction)] * 100)
    
    # Kembalikan response JSON murni (Sesuai teori Bab 2 kamu sebagai Layanan API!)
    return jsonify({
        'prediction': prediksi_teks,
        'confidence': f"{confidence:.1f}%"
    })

if __name__ == '__main__':
    import os
    # Mengambil port otomatis dari Railway, fallback ke 5000 jika dijalankan secara lokal
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)