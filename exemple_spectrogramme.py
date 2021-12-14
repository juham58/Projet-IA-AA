import librosa
import librosa.display
from pathlib import Path
import matplotlib.pyplot as plt

x, sr = librosa.load(Path.cwd()/"musicnet"/"data"/"2178.wav")

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.show()