from telebot.async_telebot import AsyncTeleBot
import telebot
import asyncio
import aiohttp
import requests
import aiofiles
import librosa
import librosa.display
import numpy as np
import matplotlib as plt
import zmq.asyncio

ctx = zmq.asyncio.Context()

socketVGG = ctx.socket(zmq.REQ)
remoteIP = "127.0.0.1"
remotePort = "9999"
socketVGG.connect(f"tcp://{remoteIP}:{remotePort}")

token = "5935648980:AAEuRoMJVCKb1RFSFdN9Z9UinxrBNnz4TYo"
bot   = AsyncTeleBot(token=token)
emotions = ["angry", "fearful", "happy", "sad", "surprised"]


async def from_oga_to_wav(filename):
    await asyncio.create_subprocess_shell(
        f"ffmpeg -i {filename}.ogg {filename}.wav"
    )


# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
async def send_welcome(message):
    await bot.reply_to(message, "Hi, this is EMOTION4ALLBOT.\nA bot used to make a large dataset of audio message with the feeling they are related to.\nIf u want to help us just send and audio message and select the feeling it is related to.")

@bot.message_handler(content_types=['voice', 'audio'])
async def processInput(message):
    markup = {}
    for emotion in emotions:
        markup[emotion] = {'callback_data': emotion}
    markup = telebot.util.quick_markup(markup, row_width=2)
    await bot.send_message(message.chat.id, "Choose one emotion:",reply_to_message_id=message.id,reply_markup=markup)


@bot.callback_query_handler(func=lambda call: True)
async def callback(call):
    if(call.message):
        text         = call.data.strip()
        chat_id      = call.message.chat.id
        messageReplyId = call.message.reply_to_message.id
        await bot.delete_message(chat_id=chat_id, message_id=call.message.id)
        await bot.send_message( chat_id, "Your audio has been processed. Thank you :)")
        # llamada a la función que envia la imagen al modelo
        file_info = await bot.get_file(call.message.reply_to_message.voice.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        socketVGG.send_multipart([
            chat_id.to_bytes((chat_id.bit_length() + 7) // 8, 'big'),
            messageReplyId.to_bytes((messageReplyId.bit_length() + 7) // 8, 'big'),
            bytes(text, encoding='utf-8'),
            emotions.index(text).to_bytes((emotions.index(text).bit_length() + 7) // 8, 'big'),
            downloaded_file
        ])
        print("Enviado")
        socketVGG.recv()


import asyncio
asyncio.run(bot.polling())