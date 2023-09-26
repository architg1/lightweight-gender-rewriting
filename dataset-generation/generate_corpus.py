import re
import spacy
import pandas as pd
from smart_convert import convert # https://github.com/googleinterns/they-them-theirs
from datasets import load_dataset
import time

nlp = spacy.load("en_core_web_sm")

MALE_PRONOUNS = ['he', 'him', 'his', 'himself']
FEMALE_PRONOUNS = ['she', 'her', 'hers', 'herself']

def is_gendered(sentence):
    """
    checks whether a sentence follows the SNAPE criteria (1 entity and 1 gender) (Sun et. al.)
    :param sentence: input sentence
    :return: true if it does and false if it does not
    """
    sentence = sentence.lower()
    contains_male = any(re.search(r'\b{}\b'.format(m_pronoun), sentence) for m_pronoun in MALE_PRONOUNS)
    contains_female = any(re.search(r'\b{}\b'.format(f_pronoun), sentence) for f_pronoun in FEMALE_PRONOUNS)
    if contains_male and not contains_female:
        return True
    elif contains_female and not contains_male:
        return True
    return False

def process_entry(text):
    """
    concatenates all sentences in a given text that fulfil follow the SNAPE criteria
    :param text: input article
    :return: concatenated article with 'EOS;' as the separator
    """
    processed_text = []
    doc = nlp(text)
    sentences = list(doc.sents)
    for sentence in sentences:
        if is_gendered(sentence.text) is True:
            processed_text.append(sentence.text)
    return 'EoS;'.join(processed_text)

def split_and_add_sentences(text, df):
    """
    fills an empty DataFrame with individual sentences from a given article as rows
    :param text: input article
    :param df: empty DataFrame
    :return: filled DataFrame
    """
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

print(f"Elapsed time: {(end_time - start_time)/60} minutes")