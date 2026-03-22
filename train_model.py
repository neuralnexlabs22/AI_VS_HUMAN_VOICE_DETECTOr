import librosa
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_features(file):
    audio, sr = librosa.load(file, duration=3)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean

X = []
y = []

valid_extensions = (".wav", ".mp3", ".aac", ".flac", ".m4a")

# Human = 0
for file in os.listdir("dataset/human"):
    if file.endswith(valid_extensions):
        path = os.path.join("dataset/human", file)
        X.append(extract_features(path))
        y.append(0)

# AI = 1
for file in os.listdir("dataset/ai"):
    if file.endswith(valid_extensions):
        path = os.path.join("dataset/ai", file)
        X.append(extract_features(path))
        y.append(1)

X = np.array(X)
y = np.array(y)

# 🔥 NEW: train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 🔥 NEW: accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))

print("Model saved!")