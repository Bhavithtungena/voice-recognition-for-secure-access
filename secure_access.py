# Import necessary libraries
import speech_recognition as sr         # For voice recognition
import pyttsx3                          # For text-to-speech
import os, subprocess                   # For executing system commands
import pywhatkit                        # For sending WhatsApp messages
import pickle, numpy as np, librosa     # For loading model and extracting audio features
import sounddevice as sd                # For recording audio
from scipy.io.wavfile import write      # For saving audio as WAV file

# Define file paths
MODEL_FILE = "svm_speaker_model.pkl"    # Trained SVM model
SCALER_FILE = "scaler.pkl"              # Scaler used during training
AUDIO_FILE = "input.wav"                # Temporary file for voice input

# Setup text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Set to female voice (optional)

# Applications that can be opened via voice command
apps = {
    "whatsapp": "start whatsapp:",
    "chrome": "start chrome",
    "notepad": "notepad",
    "command prompt": "cmd",
    "file explorer": "explorer",
    "calculator": "calc",
    "vlc": r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    "word": r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
    "excel": r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
    "visual studio code": r"C:\Users\YourUsername\AppData\Local\Programs\Microsoft VS Code\Code.exe"
}

# WhatsApp contact numbers (customize as needed)
contacts = {
    "akhil": "+919347232301",
    "pawan": "+919347628586"
}

# Convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Record and recognize speech using microphone
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎧 Listening...")
        recognizer.adjust_for_ambient_noise(source)   # Adjust to background noise
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()  # Convert audio to text
        print(f"🗣 You said: {command}")
        return command
    except:
        print("❌ Could not understand.")
        return None

# Extract MFCC features from the given WAV file
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1).reshape(1, -1)

# Verify whether the speaker is authorized using trained model
def verify_speaker(audio_path):
    features = extract_mfcc(audio_path)
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    features = scaler.transform(features)
    return model.predict(features)[0] == 1  # Return True if authorized

# Record 3 seconds of voice for authentication
def record_voice():
    fs = 44100
    seconds = 3
    print("🎙 Say your password...")
    speak("Say your password")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(AUDIO_FILE, fs, recording)
    print("✅ Voice recorded.")

# Authenticate speaker using recorded voice
def authenticate():
    speak("Please say your password.")
    record_voice()
    speak("Verifying speaker...")
    if verify_speaker(AUDIO_FILE):
        speak("Access granted.")
        return True
    else:
        speak("Access denied.")
        return False

# Handle execution of apps or WhatsApp messages
def execute_command(command):
    # WhatsApp message command
    if "whatsapp" in command and "send" in command:
        phone_number = input("Enter number with +91: ")  # Or use name lookup from `contacts` dict
        speak("What is your message?")
        message = listen()
        if message:
            pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=10)
            speak("Message sent.")
    else:
        # Open an app based on user command
        for app in apps:
            if app in command:
                speak(f"Opening {app}")
                os.system(apps[app])
                return
        speak("App not found.")

# Main program starts here
if __name__ == "__main__":
    if authenticate():  # Authenticate user by voice
        while True:
            cmd = listen()
            if cmd:
                execute_command(cmd)
