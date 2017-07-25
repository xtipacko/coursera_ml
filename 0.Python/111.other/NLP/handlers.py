#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import Filters
import io
import re
import logging
from yandexnlr import YandexNLR

#audio/ogg;codecs=opus
#audio/x-speex  
# def ifNullmakeempty
class BotHandler:
    def __init__(self, dbfile):
        self.all_handlers = [ CommandHandler('start',       self.start),
                              MessageHandler(Filters.voice, self.audio_handler) ] # check if there is smth better 
        self.silencemode = False


    def audio_handler(self, bot, update):
        print('audio_handler called')
        if self.silencemode:
            print('silence')
            return
        if not update.message.voice:
            print('no voice')
            return
        username   = update.message.from_user.username
        firstname  = update.message.from_user.first_name
        lastname   = update.message.from_user.last_name
        
        print(f'{firstname} {lastname} ({username}) sent voice message')
        
        ouput_stream = io.BytesIO()
        file_id = update.message.voice.file_id
        file_size = update.message.voice.file_size
        print(file_size)
        print(file_id)
        newFile = bot.getFile(file_id)
        print(newFile)
        newFile.download(out=ouput_stream) # download(custom_path=None, out=None, timeout=None)
        ouput_stream.seek(0)
        with open(f'audio/{file_id}.oga', 'bw+') as f:
            f.write(ouput_stream.read())
        ouput_stream.seek(0)
        text_from_speech = YandexNLR(ouput_stream, fmt='audio/ogg;codecs=opus', length=file_size) #;codecs=opus
        print(text_from_speech)
        ouput_stream.close()

        if text_from_speech:
            bot.sendMessage(chat_id=update.message.chat_id, text=f'{firstname} {lastname} ({username}) Сказал:\n{text_from_speech}')
        
    def start(self, bot, update):
        # msg =  ('Я я бот, я я я бот я бот,я бооот парам парам я боот парам парам')
        # bot.send_message(chat_id=update.message.chat_id, text=msg)
        pass