from flask import Flask, render_template, request
import pickle
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import glob

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

# -----------------------------
# Feature Extraction (FAST)
# -----------------------------
def extract_features(file):
    audio, sr = librosa.load(file, duration=3)  # 🔥 limit duration
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# -----------------------------
# Waveform Generator (FAST + CLEAN)
# -----------------------------
def save_waveform(file_path):

    # 🔥 delete old waveform images
    for file in glob.glob("static/waveform*.png"):
        try:
            os.remove(file)
        except:
            pass

    audio, sr = librosa.load(file_path, duration=3)

    plt.figure(figsize=(8, 3))
    plt.plot(audio)

    filename = "waveform.png"  # fixed name (no cache issue after refresh)
    filepath = os.path.join("static", filename)

    plt.savefig(filepath)
    plt.close()

    return filename

# -----------------------------
# Main Route
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():

    result = "Upload a file and click Analyze"
    waveform = None

    if request.method == "POST":

        if 'audio' not in request.files:
            return render_template("index.html", result="No file uploaded")

        file = request.files['audio']

        if file.filename == '':
            return render_template("index.html", result="No file selected")

        filepath = "temp_" + file.filename
        file.save(filepath)

        try:
            # Extract features
            features = extract_features(filepath)

            # Prediction
            prediction = model.predict([features])

            # Confidence
            probs = model.predict_proba([features])
            confidence = np.max(probs) * 100

            if prediction[0] == 0:
                result = f"Human Voice ({confidence:.2f}%)"
            else:
                result = f"AI Generated Voice ({confidence:.2f}%)"

            # Generate waveform
            waveform = save_waveform(filepath)

        except Exception as e:
            result = f"Error: {str(e)}"

        finally:
            # delete temp file
            if os.path.exists(filepath):
                os.remove(filepath)

    return render_template("index.html", result=result, waveform=waveform)


# 🔥 threaded = faster multiple requests
if __name__ == "__main__":
    app.run(debug=True, threaded=True)