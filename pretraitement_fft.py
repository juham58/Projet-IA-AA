import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import csv

def freq_to_idx(frequences):
    list_idx = []
    list_freq = [i*200 for i in range(0,11)]
    for i in list_freq:
        idx = (np.abs(frequences-i)).argmin()
        list_idx.append(idx)
    return list_idx

def wav_to_10(wav_dir):

    
    sampFreq, sound = wavfile.read(wav_dir)
    
    sound = sound / 2.0**6
    
    fft_spectrum = np.fft.rfft(sound)
    freq = np.fft.rfftfreq(sound.size, d=1./sampFreq)
    
    fft_spectrum_abs = np.abs(fft_spectrum)
    
    list_idx = freq_to_idx(freq)
    
    list_mean = []
    for i in range(len(list_idx)-1):
        mean = np.mean(fft_spectrum_abs[list_idx[i] : list_idx[i+1]],dtype=np.float32)
        list_mean.append(mean)
    
    '''
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (9, 7)
    plt.plot(freq[:1000000], fft_spectrum_abs[:1000000])
    plt.title('Transformée de Fourier du fichier 1727.wav')
    plt.xlabel("Fréquence [Hz]")
    plt.ylabel("Amplitude")
    plt.show()
    
    
    plt.bar([100 + i*200 for i in range(0,10)], list_mean, width=195)
    plt.title('Transformée de Fourier du fichier 1727.wav moyenné par bond de 200 [Hz]')
    plt.xlabel("Fréquence [Hz]")
    plt.ylabel("Amplitude")

    plt.plot()
    '''

    return list_mean

'''
list_mean = wav_to_10('data/musicnet/musicnet/train_data/1727.wav')
'''




file = 'data/musicnet_metadata.csv'
csv_file = open(file, mode='r')
csv_reader = csv.DictReader(csv_file)

data = []
target = []

for csv in csv_reader:
    print(csv['id'])
    list_mean = wav_to_10('data/musicnet/musicnet/train_data/{0}.wav'.format(csv['id']))
    data.append(list_mean)
    
    if csv['ensemble'] == 'Solo Piano':
        target.append(1)
    elif csv['ensemble'] != 'Solo Piano':
        target.append(0)
    


data_dict = {'data': data, 'target':target}

np.save('pretrained_data', data_dict, allow_pickle=True)


































