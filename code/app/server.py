import json
import logging
import tornado.web
import os
import sys
import pandas as pd
import numpy as np
import pickle
import h5py
from keras.models import load_model
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
from tornado.options import define, options
from tornado import gen
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from preprocess_request import clean_text,data_lstm
from keras.models import load_model
sys.path.append(os.path.realpath('../'))
sys.path.append(os.path.realpath('../../'))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')
logger = logging.getLogger(__name__)

define('port', default=9000, help='port to listen on')

class IndexHandler(RequestHandler):
    def get(self):
        self.render('../../templates/index.html')

class Predict(RequestHandler):
    def post(self):
        text = self.get_argument('text', None)
        text = clean_text(text)
        
        try :
            models
        except Exception:
            models = self.load_models()
            logger.info('Loading Models')
        
        tfidf_text = models['tfidf'].transform([text])
        logger.info('Making Prediction')
        text_lstm = data_lstm(tfidf_text)
        predict_text = models['lstm'].predict(text_lstm)
        result = models['label'].inverse_transform(np.argmax(predict_text, axis=1))
        logger.info(result)

        self.render('../../templates/results.html',data=result[0])

    def load_models(self):
        models = dict()
        path_models = '../../models/'

        models['tfidf'] = pickle.load(open(path_models+'tfidf.pickle', 'rb'))
        models['lstm'] = load_model(path_models+"lstm.h5")
        models['label'] = pickle.load(open(path_models+'label.pickle','rb'))
        
        return models

def make_app():
    urls = [("/",IndexHandler),("/results",Predict)]
    return Application(urls, debug=True)



if __name__ == '__main__':
    app = make_app()
    app.listen(options.port)
    logger.info('Serving Runing')
    logger.info('http://localhost:9000/')
    IOLoop.current().start()
    



        