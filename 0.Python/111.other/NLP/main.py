#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

# import logging
# import logging.handlers
from telegram.ext import Updater
from handlers import BotHandler
from parameters import TOKEN, SERVER_IP, PORT, CERT, UPDATE_MODE
from parameters import KEY, LOGGING_MODE, LOG_FILENAME, DB_FILE
# Делаем ротацию лога
# loghandlers = [ logging.handlers.RotatingFileHandler(LOG_FILENAME, maxBytes=10000000, backupCount=5) ]
# # Устанаваливаем режим логирования
# if LOGGING_MODE == 'DEBUG':
#     logging.basicConfig(format='[%(asctime)s]: %(message)s', level=logging.DEBUG, 
#                         handlers=loghandlers) # can set - datefmt='%d.%m.%Y %H:%M:%S'
# else:
#     logging.basicConfig(format='[%(asctime)s]: %(message)s', level=logging.WARNING, 
#                         handlers=loghandlers)




def main():
    print('CUSTOM. Main module started')
    updater = Updater(token=TOKEN)
    dispatcher = updater.dispatcher
    bothandler = BotHandler(DB_FILE)
    #Регистрируем call-back-и в dispatcher-е
    for handler in bothandler.all_handlers:
        dispatcher.add_handler(handler)   

    updater.start_polling() 
    try:
        pass
        # updater.start_webhook(listen      =  SERVER_IP,
        #                       port        =  PORT,
        #                       url_path    =  TOKEN,
        #                       key         =  KEY,
        #                       cert        =  CERT,
        #                       webhook_url =  f'https://{SERVER_IP}:{PORT!s}/{TOKEN}')
    except:
        #to-do: write trace to log file
        print('CUSTOM. Caught exception in main module')
    finally:
        pass
        #bothandler.close()
        #to-do: close, and clean...



if __name__ == '__main__':
    main()