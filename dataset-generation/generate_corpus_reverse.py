import re
import spacy
import pandas as pd
from datasets import load_dataset
import random
from gfrwriter.en.manipulator import EnglishRuleBasedReverser, EnglishNormalizer # https://github.com/textshuttle/exploiting-bias-to-debias

nlp = spacy.load("en_core_web_sm")

they_forms = r"\b(T|t)(hey|hem|heir|heirs|hemselves)\b"
gender_neutral_forms = r"(chair|spokes|business|sales) ?(persons?|people)|anchors?|members? of congress|police officers?|flight attendants?|principals?|mail carriers?|firefighters?|bartenders?|cleaners?|supervisors?"
unnecessary_female_forms = r"actors?|hero(es)?|comedians?|executors?|poets?|ushers?|authors?|boss(es)?|waiters?"
generic_man_forms = r"average (person|people)|best (person|people) for the job|lay(person|people)|husband and wife|humankind|human-made|skillful|first-year student"
pattern = f"({they_forms}|{gender_neutral_forms}|{unnecessary_female_forms}|{generic_man_forms})"

def process_entry(text): # for each entry
    processed_text = []
    doc = nlp(text)
    sentences = list(doc.sents)
    for sentence in sentences: # for each sentence
        match = re.findall(pattern, sentence.text)
        if match:
            processed_text.append(sentence.text)  # then add it back
    return 'EoS;'.join(processed_text) # return all sentences that fit our criteria

def split_and_add_sentences(text, df):
    sentences = text.split("EoS;")
    for sentence in sentences:
        df.loc[len(df)] = [sentence.strip()]

reverser = EnglishRuleBasedReverser()
normalizer = EnglishNormalizer('') 
def convert(unbiased_text):
    
    normalized_text, _ = normalizer.normalize(unbiased_text)
    _, _, biased_text_female, biased_text_male = reverser.reverse(unbiased_text, normalized_text)

    chance = random.random()
    if chance <= 0.75: # male form 75% of the times
        return biased_text_male
    else:
        return biased_text_female

# Load dataset
dataset = load_dataset("wikipedia", "20220301.simple") # dataset version
dataset = dataset['train']
dataset = pd.DataFrame(dataset)

# Process each entry to filter out sentences
dataset['text'] = dataset['text'].apply(process_entry)

# Generate the real unbiased corpus
corpus = pd.DataFrame(columns=["unbiased"])
dataset['text'].map(lambda x: split_and_add_sentences(x, corpus))
corpus.reset_index(drop=True, inplace=True)

# Generate the artificial biased corpus
corpus['biased'] = corpus['unbiased'].apply(convert)
print('Number of sentences: ', len(corpus))
print(corpus.head(n=20))

# Save the corpus
corpus.to_csv('/Users/architg/Documents/GitHub/final-year-project/data/wikipedia_corpus_reverse.csv', index=False)
print('Corpus generated!')