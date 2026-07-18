import string
from documents import DOCUMENTS

def clean_words(text):
    text = text.lower()
    for punct in string.punctuation:
        text = text.replace(punct, "")
    return set(text.split())

def retrieve(query, documents=DOCUMENTS):
    query_words = clean_words(query)    
    best_doc = None
    best_score = 0
    
    for doc in documents:
        doc_words = clean_words(doc)
        overlap = len(query_words.intersection(doc_words))
        
        if overlap > best_score:
            best_score = overlap
            best_doc = doc
    
    return best_doc, best_score

