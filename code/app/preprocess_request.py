import re

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub('\w*\d\w*','', text)
    text = text.lower()
    token_text = text.split()
    token_text = [word for word in token_text if len(word)>3]

    text_clean = " ".join(token_text)
    return text_clean

def data_lstm(tfidf):
    
    text_lstm = tfidf.toarray().reshape(tfidf.shape[0], 1, tfidf.shape[1])
    return text_lstm

