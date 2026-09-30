"""Speech emotion recognition using MFCC features and a CNN-LSTM model.

Run after installing dependencies and placing RAVDESS recordings in data/ravdess/.
This script classifies emotions; it does not diagnose depression.
"""

# # Speech Emotion Recognition
#
# An MFCC-based CNN–LSTM experiment using the RAVDESS speech dataset. This notebook performs **emotion classification only**; it does not assess or diagnose depression.
#
# The original experiment used MFCC extraction, noise/pitch augmentation, a CNN–LSTM model, and an 80/20 split. For a more reliable demonstration, this version splits recordings **by actor before augmentation** so the same speaker and derived clips do not cross the training/test boundary. Its metrics therefore should not be presented as reproducing the original reported result without running the experiment.

from pathlib import Path
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

SEED = 11
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

DATASET_DIR = Path('data/ravdess')  # Actor_01/, Actor_02/, ...
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# ## 1. Discover recordings and split by actor

records = []
for file in sorted(DATASET_DIR.glob('Actor_*/*.wav')):
    parts = file.stem.split('-')
    if len(parts) < 7:
        continue
    emotion_id, actor_id = int(parts[2]), int(parts[6])
    if 1 <= emotion_id <= 8:
        records.append((file, emotion_id - 1, actor_id))

if not records:
    raise FileNotFoundError('No RAVDESS speech WAV files found in data/ravdess/Actor_*/')

groups = [r[2] for r in records]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(records, groups=groups))
train_records = [records[i] for i in train_idx]
test_records = [records[i] for i in test_idx]
print(f'Train recordings: {len(train_records)}, test recordings: {len(test_records)}')
assert not ({r[2] for r in train_records} & {r[2] for r in test_records})

# ## 2. Audio processing and MFCC extraction

TARGET_SHAPE = (20, 104)

def extract_mfcc(path, augment=False):
    audio, sr = librosa.load(path, duration=2.4, offset=0.6)
    if augment:
        choice = random.choice(['original', 'noise', 'pitch'])
        if choice == 'noise':
            noise_scale = 0.035 * random.random() * np.max(np.abs(audio))
            audio = audio + noise_scale * np.random.normal(size=len(audio))
        elif choice == 'pitch':
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.7)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    return mfcc.astype('float32') if mfcc.shape == TARGET_SHAPE else None

def make_dataset(items, augment=False):
    features, labels = [], []
    for path, label, _ in items:
        versions = 3 if augment else 1
        for _ in range(versions):
            mfcc = extract_mfcc(path, augment=augment)
            if mfcc is not None:
                features.append(mfcc.T[..., np.newaxis])  # (104, 20, 1)
                labels.append(label)
    if not features:
        raise ValueError('No suitable MFCC samples found; check dataset files and input duration')
    return np.stack(features), np.array(labels)

X_train, y_train = make_dataset(train_records, augment=True)
X_test, y_test = make_dataset(test_records, augment=False)
print(X_train.shape, X_test.shape)

# ## 3. CNN–LSTM training

model = keras.Sequential([
    layers.Input(shape=(104, 20, 1)),
    layers.TimeDistributed(layers.Conv1D(32, 3, padding='same', activation='relu')),
    layers.TimeDistributed(layers.BatchNormalization()),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='softmax'),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    X_train, y_train, batch_size=140, epochs=80, validation_split=0.15,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6,
                                          patience=5, min_lr=1e-8),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True),
    ],
)
model.save('emotion_model.keras')

# ## 4. Test-set evaluation

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {accuracy:.3f}')
pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, pred, labels=np.arange(8),
                            target_names=EMOTIONS, zero_division=0))
cm = confusion_matrix(y_test, pred, labels=np.arange(8))
ConfusionMatrixDisplay(cm, display_labels=EMOTIONS).plot(xticks_rotation=45)
plt.tight_layout()
plt.show()
