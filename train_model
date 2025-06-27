import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Define constants for directory and file paths
AUTH_DIR = "authorized_samples"        # Folder to save voice samples from authorized person (your voice)
UNAUTH_DIR = "unauthorized_samples"    # Folder to save voice samples from others (unauthorized)
MODEL_PATH = "svm_speaker_model.pkl"   # File to save the trained SVM model
SCALER_PATH = "scaler.pkl"             # File to save the StandardScaler object

# Recording settings
fs = 44100             # Sampling frequency in Hz
seconds = 3            # Duration of each recording in seconds
num_samples = 5        # Number of samples to record for each category (authorized and unauthorized)

# Function to record a voice sample and save it to a WAV file
def record_voice(filename):
    print(f"Recording {filename}... Speak now!")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')  # Start recording
    sd.wait()  # Wait until the recording is complete
    write(filename, fs, recording)  # Save the recording to the specified file
    print(f"Saved: {filename}")

# Create directory for authorized samples if it doesn't exist
os.makedirs(AUTH_DIR, exist_ok=True)

# Record voice samples from the authorized person (you)
for i in range(1, num_samples + 1):
    record_voice(f"{AUTH_DIR}/sample_{i}.wav")

# Create directory for unauthorized samples if it doesn't exist
os.makedirs(UNAUTH_DIR, exist_ok=True)

# Record voice samples from unauthorized persons (others)
for i in range(1, num_samples + 1):
    input(f"Ask someone else to speak for sample {i}. Press Enter to record...")
    record_voice(f"{UNAUTH_DIR}/sample_{i}.wav")

# Function to extract features from an audio file using MFCCs
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)               # Load audio with librosa at 22050 Hz
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)       # Extract 13 MFCC features
    return np.mean(mfcc, axis=1)                             # Return the average of each MFCC across time

# Lists to store training data and labels
X_train, y_train = [], []

# Process authorized samples: label = 1
for file in os.listdir(AUTH_DIR):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(AUTH_DIR, file))
        X_train.append(features)
        y_train.append(1)

# Process unauthorized samples: label = 0
for file in os.listdir(UNAUTH_DIR):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(UNAUTH_DIR, file))
        X_train.append(features)
        y_train.append(0)

X_train = np.array(X_train)  # Convert feature list to NumPy array

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train an SVM classifier with a linear kernel
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Save the trained model and scaler to disk
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model trained and saved.")
