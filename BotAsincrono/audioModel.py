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
import subprocess
import aiofiles

model                = keras.models.load_model("../Model/model-laura-ad-hoc-2.h5")
validation_generator = tf.keras.utils.image_dataset_from_directory("data", color_mode="rgb",image_size=(128,157))


ctx = zmq.asyncio.Context()
       
socket = ctx.socket(zmq.REP)
socket.bind("tcp://*:9999")

async def recv_and_process():
    while True:
        msg = await socket.recv_multipart()
        await process(msg)
        socket.send(b"")

async def process(msg):
    chat_id = int.from_bytes(msg[0], 'big')
    messageReplyId = int.from_bytes(msg[1], 'big')
    text = msg[2].decode('UTF-8')
    cEmotion = int.from_bytes(msg[3], 'big')
    try:
        with open(f"./Files/{chat_id}/{messageReplyId}.oga", 'wb') as f:
            f.write(msg[-1])
    except:
        os.mkdir(f"./Files/{chat_id}")
        with aiofiles.open(f"./Files/{chat_id}/{messageReplyId}.oga", 'wb') as f:
            await f.write(msg[-1])
        await subprocess.run(['ffmpeg', '-i',f"./Files/{chat_id}/{messageReplyId}.oga", f"./Files/{chat_id}/{messageReplyId}.wav"])
        os.remove(f"./Files/{chat_id}/{messageReplyId}.oga")
    print("Guardado")

    signal, sr = librosa.load(f'./Files/{chat_id}/{messageReplyId}.wav')
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=512, n_fft=2048)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.axis('off')
    librosa.display.specshow(power_to_db, sr=sr, cmap="magma",hop_length=512)
    path = f"./Files/{text}/{chat_id}{messageReplyId}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    await train(path, text, cEmotion)


async def train(path, text, cEmotion):
    im = PIL.Image.open(path)
    a = np.asarray(im)[:,:,:3]
    model.fit(np.expand_dims(a, axis=0), np.array([cEmotion]), validation_data=validation_generator)

asyncio.run(recv_and_process())
