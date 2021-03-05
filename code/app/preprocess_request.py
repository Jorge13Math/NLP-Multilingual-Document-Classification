import re


def clean_text(text):
    """
    Clean text from request
    :param text: text from request
    :return: clean text
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.lower()
    token_text = text.split()
    token_text = [word for word in token_text if len(word) > 3]

    text_clean = " ".join(token_text)
    return text_clean


def data_lstm(tfidf):
    """
    Transform data to the input model
    :param tfidf: text format in tfidf to transform as input model
    :return:
    """
    
    text_lstm = tfidf.toarray().reshape(tfidf.shape[0], 1, tfidf.shape[1])
    return text_lstm
