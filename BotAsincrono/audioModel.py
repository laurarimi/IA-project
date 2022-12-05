import tensorflow as tf
import keras
import zmq
import zmq.asyncio
import asyncio
import librosa, librosa.display
import numpy as np
import os
import PIL
import PIL.Image
import aiofiles
from scipy.io import wavfile
import threading
import concurrent.futures


model                = keras.models.load_model("./Model/modelNoImage.h5")

th_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
lock = threading.Lock()

ctx = zmq.asyncio.Context()
       
socket = ctx.socket(zmq.REP)
socket.bind("tcp://*:9999")

async def recv_and_process():
    while True:
        msg = await socket.recv_multipart()
        loop = asyncio.get_event_loop()
        await asyncio.wait([loop.run_in_executor(th_executor, process, msg)])
        await socket.send_string("")

            

def fragmentation(file):    
    fs, signal = wavfile.read(file) 
    signal_len = len(signal) 
    segment_size_t = 3 # segment size in seconds 
    segment_size = segment_size_t * fs # segment size in samples # Break signal into list of segments in a single-line Python code 
    segments = np.array([signal[x:x + segment_size] for x in np.arange( 0 , signal_len, segment_size)]) # Save each segment in a seperate filename 
    for iS, s in enumerate(segments): 
        wavfile.write(file,fs, (s))
    print("Guardado")

desiredLength = 16000 * 8

def generateMELSpectrogram(file, emotion):
    signal, sr = librosa.load(file)    
    padding = desiredLength - len(signal) #cantidad de tiempo que falta
    if(padding < 0):
        return None
    signal = np.concatenate([np.zeros((padding)), signal])
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=512, n_fft=2048)
    spectrogram = np.abs(mel_signal)
    power_to_db = np.expand_dims(librosa.power_to_db(spectrogram, ref=np.min), axis=2)
    return power_to_db


def process(msg):
    chat_id = int.from_bytes(msg[0], 'big')
    messageReplyId = int.from_bytes(msg[1], 'big')
    text = msg[2].decode('UTF-8')
    cEmotion = int.from_bytes(msg[3], 'big')
    if(f"{chat_id}" not in os.listdir("./Files/")):    
        os.mkdir(f"./Files/{chat_id}")
    with open(f"./Files/{chat_id}/{messageReplyId}.oga", 'wb') as f:
            f.write(msg[-1])
    
    os.system(f'ffmpeg -i ./Files/{chat_id}/{messageReplyId}.oga ./Files/{text}/{messageReplyId}.wav')
    os.remove(f"./Files/{chat_id}/{messageReplyId}.oga")
    fragmentation((f"./Files/{text}/{messageReplyId}.wav"))
    matrix = generateMELSpectrogram(f"./Files/{chat_id}/{messageReplyId}.wav", text)
    np.save(f'./Files/{text}/{messageReplyId}', matrix)
    if(matrix is not None):
        return train(matrix, cEmotion)


def train(matrix, cEmotion):
    lock.acquire()
    model.fit(np.expand_dims(matrix, axis=0), np.array([cEmotion]))
    lock.release()

asyncio.run(recv_and_process())
