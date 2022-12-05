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
import struct

ctx = zmq.asyncio.Context()

socketVGG = ctx.socket(zmq.REQ)
# Esto es para el bot que se entrena
#remoteIP = "158.42.176.208"
remoteIP = "localhost"
remotePort = "9999"
socketVGG.connect(f"tcp://{remoteIP}:{remotePort}")

# Esto es para el bot que predice
remoteIPPred    = "localhost"
remotePortPred  = "9999"
socketPred      = ctx.socket(zmq.REQ)
socketPred.connect(f"tcp://{remoteIPPred}:{remotePortPred}")


token = "5935648980:AAEuRoMJVCKb1RFSFdN9Z9UinxrBNnz4TYo"
bot   = AsyncTeleBot(token=token)
emotions = np.array(["angry", "fearful", "happy", "sad", "surprised"])
emotionsSize ={
    "angry" : 384,
    "fearful" : 192,
    "happy" : 480,
    "sad" : 192,
    "surprised" : 192
}
threshold = 20
to_predict = list()


info = '''
Hola, soy EMOTION4ALLBOT.

Soy un bot diseñado por los alumnos Carlos March, Laura Rivero y Mario Rico de la Universitat Politécnica de Valencia (UPV) para generar una base de datos de audios junto con el sentimiento que tienen asociado, con el fin de poder usar estos datos para entrenar una inteligencia artificial que pueda predecir el sentimiento asociado a un audio y poder ayudar a las personas que sufren el transtorno de espectro autista a entender el mundo que los rodea.

Siempre que quieras ayudarnos solo tienes que seguir los siguientes tres pasos:

1. Envia el mensaje \status (o pulsa sobre él), para obtener la lista de los sentimientos de los cuales necesitamos audios en ese momento. Como se reciben muchos audios necesitamos tener más o menos la misma cantidad de audios de cada sentimiento, por eso limitamos los sentimientos que necesitamos. 

2. Envia un audio, con una duración máxima de 7 segundos. Normalmente frases con más de tres o cuatro segundos pueden tener más de un sentimiento relacionado y eso puede dificultar el proceso de entrenamiento de la inteligencia artificial.

3. Selecciona de la lista de opciones que se te enviarán el sentimiento que más se acerca al audio. Si te has equivocado, el audio no es correcto o no se ajusta a ningún sentimiento, no te preocupes, hay una opción extra llamada "No procesar" que permite descartar ese audio.

Desde el equipo te damos las gracias por ayudarnos y sería de mucha ayuda que compartieses esta herramienta :)
'''    


async def from_oga_to_wav(filename):
    await asyncio.create_subprocess_shell(
        f"ffmpeg -i {filename}.ogg {filename}.wav"
    )


# Handle '/start' and '/help' commands
@bot.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    await bot.reply_to(message, info)

# Handle '/predict' command
@bot.message_handler(commands=['predict'])
async def predict_command(message):
    markup = telebot.types.ForceReply()
    msg = await bot.reply_to(message,"Responde a este mensaje con un audio",reply_markup=markup)
    to_predict.append(msg.id)


# Handle 'status' command
@bot.message_handler(commands=['status'])
async def showStatus(message):
    minEmotioSize = emotionsSize[min(emotionsSize, key=emotionsSize.get)]
    res = []
    for emotion in emotions:
        if emotionsSize[emotion] < minEmotioSize + threshold:
            res.append(emotion)
    await bot.reply_to(message, f"Las emociones que nos hacen falta en este momento son: {', '.join(res)}.\n Envianos una porfa :)")
        

# Handle the reception of an audio and voice message
@bot.message_handler(content_types=['voice', 'audio'])
async def processInput(message):
    chat_id        = message.chat.id
    if(message.reply_to_message):
        messageReplyId = message.reply_to_message.id
        if(messageReplyId and messageReplyId in to_predict):
            to_predict.remove(messageReplyId)        
            file_info = await bot.get_file(message.voice.file_id)
            downloaded_file = await bot.download_file(file_info.file_path)
            await socketPred.send_multipart([
                chat_id.to_bytes((chat_id.bit_length() + 7) // 8, 'big'),
                messageReplyId.to_bytes((messageReplyId.bit_length() + 7) // 8, 'big'),
                downloaded_file
            ])
            msg = await socketPred.recv_multipart()
            await bot.reply_to(message, f"Predicción:{emotions[int.from_bytes(msg[0], 'big')]}\nProbabilidad:{0:.2f}".format(struct.unpack('f',msg[1])[0]*100))
    else:
        markup = {}
        minEmotioSize = emotionsSize[min(emotionsSize, key=emotionsSize.get)]
        for emotion in emotions:
            if emotionsSize[emotion] < minEmotioSize + threshold:
                markup[emotion] = {'callback_data': emotion}
        markup['No Procesar'] = {'callback_data': "No procesar"}
        markup = telebot.util.quick_markup(markup, row_width=2)
        await bot.send_message(message.chat.id, "Escoge una emoción:",reply_to_message_id=message.id,reply_markup=markup)

# Handle the feeling selected by the user
@bot.callback_query_handler(func=lambda call: True)
async def callback(call):
    if(call.message != "No procesar"):
        text         = call.data.strip()
        chat_id      = call.message.chat.id
        messageReplyId = call.message.reply_to_message.id
        await bot.delete_message(chat_id=chat_id, message_id=call.message.id)
        await bot.send_message( chat_id, "Tu audio ha sido procesado. Gracias <3 :)")
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
        await socketVGG.recv_string()
        emotionsSize[text] += 1
    else:
        await bot.send_message( chat_id, "Intentalo de nuevo :)")

# Initialize the bot
import asyncio
asyncio.run(bot.polling())