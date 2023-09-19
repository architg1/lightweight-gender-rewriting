import re
import spacy
import pandas as pd
from smart_convert import convert
from datasets import load_dataset
import time

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
    return 'EoS;'.join(processed_text) # return all sentences that fit our criteria

def split_and_add_sentences(text, df):
    sentences = text.split("EoS;")
    for sentence in sentences:
        df.loc[len(df)] = [sentence.strip()]

start_time = time.time()
# Load dataset
dataset = load_dataset("wikipedia", "20220301.simple") # dataset version
dataset = dataset['train']
dataset = pd.DataFrame(dataset)
print('Number of articles: ', len(dataset))
print(dataset.head(n=5))

# Process each entry to filter out sentences
dataset['text'] = dataset['text'].apply(process_entry)

# Generate the biased corpus
corpus = pd.DataFrame(columns=["biased"])
dataset['text'].map(lambda x: split_and_add_sentences(x, corpus))
corpus.reset_index(drop=True, inplace=True)

# Convert biased text into unbiased text
corpus['unbiased'] = corpus['biased'].apply(convert)
print('Number of sentences: ', len(corpus))
print(corpus.head(n=20))

# Save the corpus
corpus.to_csv('/Users/architg/Documents/GitHub/final-year-project/data/wikipedia_corpus.csv', index=False)
print('Corpus generated!')
end_time = time.time()

print(f"Elapsed time: {(end_time - start_time)/60} seconds")