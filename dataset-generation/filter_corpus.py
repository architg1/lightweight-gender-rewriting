import pandas as pd
import numpy as np

def count_words(text):
    words = text.split()
    return len(words)

# Load original corpus
df = pd.read_csv('/Users/architg/Documents/GitHub/final-year-project/data/wikipedia_corpus.csv')

df = df.dropna() # drop empty rows
df = df[df['biased'].apply(lambda x: count_words(x) <= 20)] # drop sentences with >20 words
df.reset_index(drop=True, inplace=True) # reset index

# Save filtered corpus
df.to_csv('/Users/architg/Documents/GitHub/final-year-project/data/wikipedia_corpus_filtered.csv', index=False)