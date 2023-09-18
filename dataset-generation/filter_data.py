import re
import spacy
import pandas as pd
from smart_convert import convert
from datasets import load_dataset

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
        df.loc[len(df)] = sentence.strip()

# Load dataset
dataset = load_dataset("wikipedia", "20220301.simple")
dataset = dataset['train']
dataset = pd.DataFrame(dataset)
print(dataset.head(n=5))

print('Number of articles: ', len(dataset))
dataset['text'] = dataset['text'].apply(process_entry)

# Generate the biased corpus
corpus = pd.DataFrame(columns=["biased"])
dataset.map(split_and_add_sentences)
corpus.reset_index(drop=True, inplace=True)
pd.display(corpus)

# Generate the unbiased corpus
corpus['unbiased'] = corpus['biased'].apply(convert)

# Save the corpus
corpus.to_csv('wikipedia_corpus', index=False)