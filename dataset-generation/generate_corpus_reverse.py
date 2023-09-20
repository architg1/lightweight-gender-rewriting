import re
import spacy
import pandas as pd
from datasets import load_dataset
from smart_convert_reverse import convert_reverse

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

# Load dataset
dataset = load_dataset("wikipedia", "20220301.simple") # dataset version
dataset = dataset['train']
dataset = pd.DataFrame(dataset)

# Process each entry to filter out sentences
dataset['text'] = dataset['text'].apply(process_entry)