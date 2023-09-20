import re
import torch
from string import punctuation

# SpaCy: lowercase is for dependency parser, uppercase is for part-of-speech tagger
import spacy
from spacy.symbols import nsubj, nsubjpass, conj, poss, obj, iobj, pobj, dobj, VERB, AUX, NOUN
from spacy.tokens import Token, Doc

# load SpaCy's "en_core_web_sm" model
# English multi-task CNN trained on OntoNotes
# Assigns context-specific token vectors, POS tags, dependency parse and named entities
# https://spacy.io/models/en
import en_core_web_sm

nlp = en_core_web_sm.load()

# look at scripts/prepare.sh, gfrwriter/cli/prepare.py, and gfrwriter/en/manipulator.py
def convert_reverse(sentence: str) -> str:
    pass