# voice-recognition-for-secure-access
A speaker recognition system using MFCC and SVM that verifies your voice and blocks unauthorized access.
# 🔐 Voice Recognition for Secure Access using MFCC + SVM

This is a secure voice-controlled system that verifies if the speaker is an authorized user before allowing access to applications or sending messages. It uses **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction and **SVM (Support Vector Machine)** for speaker classification.

---

## 🎯 Project Goal

To allow **only one specific person (you)** to access system applications or execute voice commands by verifying voice identity. If any other voice tries to interact, access is denied and can be optionally logged or alerted.

---

## 🧠 How It Works

1. You record your voice (`authorized_samples/`) and collect a few samples from others (`unauthorized_samples/`).
2. MFCC features are extracted from all samples.
3. An SVM classifier is trained to distinguish your voice from others.
4. A secure access interface allows only your voice to control the system or send WhatsApp messages.

---

## 🔧 Training the Model

Run this command to start training:

```bash
python train_model.py
