#Import Counter and tokenize for the bag of words model to count the number of words for intent classification
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
import re
from nltk import pos_tag
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
import spacy
import en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def preprocess_sentence(input_sentence):
    input_sentence = input_sentence.lower()
    #removes all characters in the input_sentence that are not(^) letters, numbers or whitespace
    input_sentence = re.sub(r'[^\w\s]','',input_sentence)
    tokens = word_tokenize(input_sentence)
    input_sentence = [i for i in tokens if not i in stop_words]
    return input_sentence

def compare_overlap(user_message, possible_responses):
    similar_words = 0
    for tokens in user_message:
        if tokens in possible_responses:
            similar_words += 1
    return similar_words

def extract_nouns(tagged_message):
    message_nouns = []
    for tokens in tagged_message:
        if tokens[1].startswith("N"):
            message_nouns.append(tokens[0])
    return message_nouns

def compute_similarity(tokens, category):
    results_list = []
    for token in tokens:
      results_list.append([token.text, category.text, token.similarity(category)])
    return results_list


    



