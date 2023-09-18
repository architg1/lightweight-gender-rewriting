from collections import Counter, defaultdict
import re
import spacy
import matplotlib.pyplot as plt
import pandas as pd

nlp = spacy.load("en_core_web_sm")

MALE_PRONOUNS = ['he', 'him', 'his', 'himself']
FEMALE_PRONOUNS = ['she', 'her', 'hers', 'herself']

def is_gendered(sentence):
    sentence = sentence.lower()
    contains_male = any(re.search(r'\b{}\b'.format(m_pronoun), sentence) for m_pronoun in MALE_PRONOUNS)
    contains_female = any(re.search(r'\b{}\b'.format(f_pronoun), sentence) for f_pronoun in FEMALE_PRONOUNS)
    if contains_male and not contains_female:
        return True
    elif contains_female and not contains_male:
        return True
    return False

def process_entry(text): # for each entry
    processed_text = []
    doc = nlp(text)
    sentences = list(doc.sents)
    for sentence in sentences: # for each sentence
        if is_gendered(sentence.text) is True: # if the sentence fulfils our criteria
            processed_text.append(sentence.text)  # then add it back
    return ';;'.join(processed_text) # return all sentences that fit our criteria

# Load dataset
dataset = load_dataset("wikipedia", "20220301.en")
dataset['text (string)'] = dataset['text (string)'].apply(process_entry)

"""
gendered_dataset = []

articles = dataset['text (string)']
for article in articles:
    sentences = nlp(article)
    for sentence in sentences:
        if is_gendered(sentence) is True:
            gendered_dataset.append(sentence)

gendered_dataset = pd.DataFrame({"gendered_sentence": gendered_dataset})
"""
