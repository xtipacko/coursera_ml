#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import requests
import random
import xmltodict
from pprint import pprint

def makelst(var):
    if type(var) != list:
        return [ var ] 
    else: 
        return var

def YandexNLR(filestream, fmt='audio/x-mpeg-3', length=0):
    print('Im here')
    random.seed(243)
    bits = random.getrandbits(128)
    user_id = f'{bits:32x}'
    user_key = 'bf18ee20-8b3d-4fde-b601-eb72974d2023'
    url = f'http://asr.yandex.net/asr_xml?uuid={user_id}&key={user_key}&topic=queries'
    file={'file': filestream}    
    headers = { #'Host'         : 'asr.yandex.net',
                'Content-Type' : fmt,
                #'Content-Length' : length
              }
    r = requests.post(url, files=file, headers=headers)
    print('sending request')
    print(r.text)
    reply = xmltodict.parse(r.text)
    pprint(reply)
    best_variant = ('', float('-inf'))
    print('choosing best_variant')
    if ('recognitionResults' in reply and
    reply['recognitionResults']['@success'] == '1' ):
        print('choosing best_variant start')
        for variant in makelst(reply['recognitionResults']['variant']):
            confidence = float(variant['@confidence'])
            print('Сравниваю уверенность')
            if  confidence > best_variant[1]:
                print(f'Хорошая уверенность {confidence}')
                best_variant = (variant['#text'], confidence)
            else:
                print(f'Плохая уверенность {confidence}')
        print('В итоге ',  best_variant[0])
        return best_variant[0]
    else:
        print('чёта пошло не так')
        return None
