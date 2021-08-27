"""Get surprisal on anaphoric word with BERT.
"""

import re
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk import word_tokenize
from collections import Counter


from torch import tensor, softmax, no_grad
from transformers import (BertTokenizer, BertForMaskedLM,
                          # pipeline,
                          )

# Load pre-trained model tokenizer (vocabulary)
BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
BERT = BertForMaskedLM.from_pretrained('bert-base-uncased')
BERT.eval()


def mask_probability(text, candidates, model=BERT, tokenizer=BERT_TOKENIZER):
    """Probabilities for candidates as replacement for [MASK] in text

    Args:
        text (str): Text containing a single [MASK]
        candidates (list of str): Candidate mask replacements
        model (TYPE, optional): language model (BERT)
        tokenizer (TYPE, optional): lm tokenizer (BERT)

    Returns:
        candidates (dict): {candidate: prob}
    """

    # Check exactly one mask
    masks = sum(np.array(text.split()) == "[MASK]")
    if masks != 1:
        raise ValueError(
            f"Must be exactly one [MASK] in text, {masks} supplied.")

    # Get candidate ids
    candidate_ids = {}
    for candidate in candidates:
        candidate_tokens = candidate.split()
        candidate_ids[candidate] = tokenizer.convert_tokens_to_ids(
            candidate_tokens)


    candidate_probs = {}

    # Loop through candidates and infer probability
    for candidate, ids, in candidate_ids.items():

        # Add a mask for each token in candidate

        candidate_text = re.sub("\[MASK\] ", "[MASK] " * len(ids), text)

        # Tokenize text
        tokenized_text = tokenizer.tokenize(candidate_text)
        mask_inds = np.where(np.array(tokenized_text) == "[MASK]")

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = tensor([indexed_tokens])

        # Predict all tokens
        with no_grad():
            outputs = model(tokens_tensor)  # token_type_ids=segments_tensors)
            predictions = outputs[0]

        # get predicted tokens
        probs = []

        for (i, mask) in enumerate(mask_inds[0]):
            # prediction for mask
            mask_preds = predictions[0, mask.item()]
            mask_preds = softmax(mask_preds, 0)
            prob = mask_preds[ids[i]].item()
            probs.append(np.log(prob))

        candidate_probs[candidate] = np.exp(np.mean(probs))

    return candidate_probs


def nth_repl(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s


### PATHS
STIMULI_PATH = "data/raw/similarity.csv"
SAVE_PATH = "data/processed/surprisals.csv"

### Read in stims
df_stims = pd.read_csv(STIMULI_PATH)
df_stims.head(5)

### Alter columns
df_stims['sentence'] = df_stims['Zeugmatic Similarity Norming Sentence']
df_stims['word'] = df_stims['CW'].str.lower()
df_stims['string'] = df_stims['String'].str.lower()

comparisons = []

with tqdm(total=len(df_stims)) as progress_bar:
    for index, row in df_stims.iterrows():

        target_word = row['string']
        sentence = row['sentence'].lower().replace(".", " .")
        ## Hand-coded
        anaphora = row['Anaphora']

        ## Remove these from stims
        if sentence[0] == "*":
            continue

        ## For now, ignore items where anaphora has multiple words
        if " " in anaphora:
            print(sentence)
            print(anaphora)
            continue

        ## Create masked sentence
        ### First, separate all words
        sentence_buffered = sentence.replace(",", " ,")
        buffered_anaphora = ' {a} '.format(a = anaphora) 

        words = word_tokenize(sentence)
        if Counter(words)[anaphora] > 1 or target_word == anaphora:
            sentence_masked = nth_repl(sentence_buffered, buffered_anaphora, " [MASK] ", 2)
        else:
            sentence_masked = sentence_buffered.replace(buffered_anaphora, " [MASK] ")

        ## Get masked probability of anaphor appearing there
        candidates = [anaphora]
        p = mask_probability(sentence_masked, candidates)



        ## add to dict
        new_dict = {
            'string': target_word,
            'word': row['word'],
            'probability': p[anaphora],
            'Similarity Norming Category': row['Similarity Norming Category']
            }

        ## Add to list
        comparisons.append(new_dict)
        ## Update progress bar
        progress_bar.update(1)


            
## Create dataframe
df_surprisals = pd.DataFrame(comparisons)
print(len(df_surprisals))

## Save data
df_surprisals.to_csv(SAVE_PATH)
