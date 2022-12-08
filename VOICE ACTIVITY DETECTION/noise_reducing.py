import noisereduce as nr
import scipy.io.wavfile as wf


def noise_reducing(filename):
    rate, data = wf.read(filename)
    data = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.4) #ajustar prop_decrease para que no sea tan agresivo
    wf.write(filename, rate, data)

noise_reducing('audio.wav')


