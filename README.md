# Early Depression Detection via Speech Emotion Recognition

A deep learning project that explores speech emotion recognition as a potential approach to monitoring emotional patterns and supporting research into early indicators of depression.

## Overview

The system analyzes audio recordings to recognize emotions through acoustic feature extraction and deep learning. It aims to reduce reliance on manual mood reporting by exploring automated speech-based emotion monitoring.

The model classifies emotions expressed in speech rather than diagnosing depression.

## System Architecture

**Audio Input → Preprocessing → MFCC Feature Extraction → CNN-LSTM → Emotion Prediction**

The system consists of the following components:

- **Audio Processing:** Loads, trims and resamples audio recordings using Librosa.
- **Feature Extraction:** Extracts Mel-Frequency Cepstral Coefficients (MFCCs) to represent acoustic characteristics.
- **Data Augmentation:** Applies noise injection and pitch shifting to increase training-data diversity.
- **Deep Learning:** Uses CNN layers for acoustic feature extraction and LSTM layers for temporal modeling.
- **Emotion Classification:** Predicts emotional categories using a softmax classification layer.
- **User Interface:** Uses Gradio to accept audio recordings and display predictions.

## Dataset

**RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song**

An emotional speech dataset containing recordings of actors expressing different emotional states.

## Model

A hybrid CNN-LSTM architecture combining convolutional feature extraction with recurrent temporal modeling.

The preprocessing pipeline generates MFCC representations of audio recordings, which are used as inputs to the neural network.

## Technologies

`Python` `TensorFlow` `Keras` `Librosa` `Scikit-learn` `NumPy` `Gradio`

## Project Scope

The implementation focuses on automatic speech emotion classification. Using emotion-recognition results to identify potential depression indicators is a proposed application and has not been clinically validated.
