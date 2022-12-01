import tensorflow as tf
import keras
import zmq
import zmq.asyncio
import asyncio
import aiofiles
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import struct

model = keras.models.load_model("../Model/model-laura-ad-hoc-2.h5")



ctx = zmq.asyncio.Context()
       
socket = ctx.socket(zmq.REP)
socket.bind("tcp://*:9999")

async def recv_and_process():
    while True:
        msg = await socket.recv_multipart()
        pred = await predict(msg)
        index = np.argmax(pred)
        await socket.send_multipart([
            index.to_bytes((index.bit_length() + 7) // 8, 'big'),
            struct.pack("f", pred[0,index])
        ])

async def predict(msg):
    chat_id = int.from_bytes(msg[0], 'big')
    messageReplyId = int.from_bytes(msg[1], 'big')
    with open(f"./temp/{chat_id}{messageReplyId}.oga", "wb") as f:
        f.write(msg[-1])
    signal, sr = librosa.load(f'./temp/{chat_id}{messageReplyId}.oga')
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=512, n_fft=2048)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.axis('off')
    librosa.display.specshow(power_to_db, sr=sr, cmap="magma",hop_length=512)
    path = f"./temp/{chat_id}{messageReplyId}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    im = PIL.Image.open(path)
    a = np.asarray(im)[:,:,:3]
    prediction = model.predict(np.expand_dims(a, axis=0))
    os.remove(f"./temp/{chat_id}{messageReplyId}.oga")
    os.remove(f"./temp/{chat_id}{messageReplyId}.png")
    return prediction

asyncio.run(recv_and_process())
