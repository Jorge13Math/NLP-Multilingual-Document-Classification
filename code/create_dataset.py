import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')


logger = logging.getLogger(__name__)


def genererate_dataframe(path):
    """
    Generate dataframe from folder data
    
    Args:
        :path: Folder's Path

    Returns:
        :df: Dataframe with text label and language
    
    """
    logger.info('Loading files')
    labels = os.listdir(path)
    languages = ['en', 'es', 'fr']
    values = {}
    data = []
    for label in labels:
        for language in languages:
            if os.path.isdir(path+label+'/'+language):
                for doc in os.listdir(path+label+'/'+language):
                    
                    f = open(path+label+'/'+language+'/'+doc, 'r', encoding='utf-8')
                    text = f.read()
                    values['Text'] = text
                    f.close()
                    values['language'] = language
                    values['label'] = label
                    data.append(values)
                    values = {}
    
    df = pd.DataFrame(data)
    logger.info('Dataframe generated')
    df.to_csv(path+'dataset_multilng.csv')
    path_csv = path+'dataset_multilng.csv'

    logger.info('Saving dataframe as csv file in this path:' + path_csv)
    return df
