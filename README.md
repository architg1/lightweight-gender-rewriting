# final-year-project
Building gender-fair rewriting models using lightweight transformers 

## dataset-generation
Generates a parallel biased-unbiased corpus from 205,328 Wikipedia articles

### Forward Augmentation
#### generate_corpus.py
* Filters for biased text (source), then uses debiasing techniques to create unbiased text (psuedo target)


### Backward Augmentation
#### generate_corpus_reverse.py
* Filters for unbiased text (target), then reverses forward augmentation  to create biased text (psuedo source)

### Shared Functions
#### filter_corpus.py
Performs post-processing on the generated corpora
