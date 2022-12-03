from scipy.io import wavfile
import noisereduce as nr
# import webrtcvad
# load data

def noisereducing(file):
    rate, data = wavfile.read(file)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(file, rate, reduced_noise)

# def audio_detection(file):
#     vad = webrtcvad.Vad()
#     vad.set_mode(1)
#     # Initialize a vad object
#     audioFile = wave.open('cero_1_1.wav')
#     framesAudio = audioFile.readframes(800)
#     #print(fraud.frames)

#     vad = webrtcvad.Vad()
#     # Run the VAD on 10 ms of silence and 16000 sampling rate 
#     sample_rate = 16000
#     frame_duration = 10  # in ms
#     for f in framesAudio :
#         # Detecting speech
#         final_frame = f.to_bytes(2,"big")* int(sample_rate * frame_duration / 1000)

#         print(f'Contains speech: {vad.is_speech(final_frame, sample_rate)}')

if __man__ == '__name__':
    noisereducing("cero_1_1.wav")
