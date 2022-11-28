token = "5935648980:AAEuRoMJVCKb1RFSFdN9Z9UinxrBNnz4TYo"

import telebot
from telebot import types
import requests
import json 
import PIL
import PIL.Image
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import os


def from_oga_to_wav(filename):
    os.system("ffmpeg -i {0}.oga  {0}.wav".format(filename))


files = {}

emotions = ["anger", "boredorm", "fear", "happiness", "sadness", "surprised"]

model = keras.models.load_model("LauraModel.h5")
bot = telebot.TeleBot(token)


@bot.message_handler(commands=["start", "help"])
def sendInfo(message):
    print('Recibido')
    bot.reply_to(message, "Hi, this is EMOTION4ALLBOT.\nA bot used to make a large dataset of audio message with the feeling they are related to.\nIf u want to help us just send and audio message and select the feeling it is related to.")

@bot.message_handler(content_types=['voice','audio'])
def processInput(message):
    print('Recibido')
    file_info = bot.get_file(message.voice.file_id)
    file = from_oga_to_wav(requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(token, file_info.file_path)).content)
    #markup = types.ForceReply()
    #sentMessage = bot.send_message(message.chat.id, "Choose one emotion:\n", reply_to_message_id=message.id, reply_markup=markup)
    # markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    # for emotion in emotions:
    #     markup.add(types.KeyboardButton(emotion))
    # sentMessage = bot.send_message(message.chat.id, "Choose one emotion:\n", reply_to_message_id=message.id, reply_markup=markup)

    # markup = types.InlineKeyboardMarkup()
    # for emotion in emotions:
    #     markup.add(types.InlineKeyboardButton(text=emotion, callback_data = emotion))
    markup = {}
    for emotion in emotions:
        markup[emotion] = {'callback_data': emotion}
    markup = telebot.util.quick_markup(markup, row_width=2)
    sentMessage = bot.send_message(message.chat.id, "Choose one emotion:",reply_to_message_id=message.id,reply_markup=markup)
    try:
        files[message.chat.id][message.id] = file
    except:
        files[message.chat.id] = {message.id : file}
    
    try:
        with open(f"./Bot/files/{message.chat.id}/{message.id}.wav", 'wb') as f:
            f.write(file)
    except:
        os.mkdir(f"./Bot/files/{message.chat.id}")
        with open(f"./Bot/files/{message.chat.id}/{message.id}.wav", 'wb') as f:
            f.write(file)
    print(message.chat.id, message.id)

@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    if(call.message):
        text         = call.data.strip()
        chat_id      = call.message.chat.id
        messageReplyId = call.message.reply_to_message.id
        print(chat_id, messageReplyId)
        file = files[chat_id][messageReplyId]        
        signal, sr = librosa.load(f'./Bot/files/{chat_id}/{messageReplyId}.wav')
        mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=512, n_fft=2048)
        spectrogram = np.abs(mel_signal)
        power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
        plt.axis('off')
        librosa.display.specshow(power_to_db, sr=sr, cmap="magma",hop_length=512)
        path = f"./Bot/Dataset/{text}/{chat_id}{messageReplyId}.png"
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        os.remove(f'./Bot/files/{chat_id}/{messageReplyId}.wav')
        im = PIL.Image.open(path)
        a = np.asarray(im)[:,:,:3]
        print(model.fit(np.expand_dims(a, axis=0), np.array([emotions.index(text)])))
        bot.delete_message(chat_id=chat_id, message_id=call.message.id)
        bot.send_message( chat_id, "Your audio has been processed. Thank you :)")
    
''' 
@bot.message_handler(func=lambda message: message.reply_to_message is not None)
def classificate(message):
    text         = message.text.lower().strip()
    chat_id      = message.chat.id
    messageReplyId = message.reply_to_message.id
    if(text in emotions):
        file = files[chat_id][messageReplyId]        
        signal, sr = librosa.load(f'./Bot/files/{chat_id}/{messageReplyId}.oga')
        mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=512, n_fft=2048)
        spectrogram = np.abs(mel_signal)
        power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
        plt.axis('off')
        librosa.display.specshow(power_to_db, sr=sr, cmap="magma",hop_length=512)
        path = f"./Bot/Dataset/{text}/{chat_id}{messageReplyId}.png"
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        os.remove(f'./Bot/files/{chat_id}/{messageReplyId}.oga')
        im = PIL.Image.open(path)
        a = np.asarray(im)[:,:,:3]
        print(model.fit(np.expand_dims(a, axis=0), np.array([emotions.index(text)])))
    else:
        markup = types.ForceReply()
        bot.reply_to(message.reply_to_message, "Choose one emotion:\n" + ", ".join(emotions), reply_markup=markup)
'''



bot.infinity_polling()
