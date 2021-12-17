import librosa
import librosa.display
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

x, sr = librosa.load(Path.cwd()/"musicnet"/"data"/"2298.wav", duration=2.0)

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.show()

#S = librosa.samples_like(X)

#print(S.shape)
#print(S)
#print(np.angle(X).shape)
#print(np.abs(X[:,1]))
#print(np.abs(X))
#print(np.argmax(np.abs(X),axis=1))
#print(librosa.note_to_midi(librosa.hz_to_note(np.argmax(np.abs(X),axis=1)))[:87])
#print(sr)

#print(np.min(Xdb[0]))