import pandas as pd
import numpy as np

# Load original corpus
df = pd.read_csv('/Users/architg/Documents/GitHub/final-year-project/data/wikipedia_corpus.csv')

df = df.dropna()

# Save filtered corpus
df.to_csv('/Users/architg/Documents/GitHub/final-year-project/data/wikipedia_corpus_filtered.csv', index=False)