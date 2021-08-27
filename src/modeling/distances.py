"""Get cosine distances with BERT, using transformers package."""


import torch
from transformers import BertTokenizer, BertModel


from scipy.spatial.distance import cosine
from nltk import word_tokenize
from collections import Counter


def get_tokens(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [1] * len(tokenized_text)
    return tokenized_text, indexed_tokens, segment_ids


def convert_to_tensor(indexed_tokens, segment_ids):
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
    return tokens_tensor, segments_tensors


def embed_sentence(tokens_tensor, segments_tensors):
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
    return outputs


def get_layers(outputs):
    """Assumes outputs contains hidden states.

    Then transforms into tensors."""
    hidden_states = outputs[2]

    # Transform into single tensor
    token_embeddings = torch.stack(hidden_states, dim=0)
    ## Remove "batch" layer b/c we just have one sentence
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    ## Rearrange layers, now: 1) token; 2) layer; 3) unit
    token_embeddings = token_embeddings.permute(1,0,2)

    return token_embeddings



def get_embeddings_for_sentence(sentence):
    # Tokenize
    tt, indexed_tokens, segment_ids = get_tokens(sentence, tokenizer)
    # Convert to tensors
    tokens_tensor, segments_tensors = convert_to_tensor(indexed_tokens, segment_ids)
    # Get embeddings
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
    # Reshape embeddings
    token_embeddings = get_layers(outputs)

    return tt, token_embeddings

def get_embeddings_for_word(target_word, tt, token_embeddings, sentence, anaphor = False):
    """Get embeddings for target word."""
    if target_word not in tt:
        og_index = find_og_index(target_word, sentence)
        return get_wordpiece_embeddings(og_index, tt, token_embeddings)

    ## Get all indices of target word
    indices = [i for i, x in enumerate(tt) if x == target_word]
    if anaphor:
        target_index = indices[-1]
    else:
        target_index = indices[0]
    # Index of target word
    # target_index = tt.index(target_word)

    return token_embeddings[target_index]


### For getting OOV tokens
def find_og_index(target, sentence, anaphor=False):
    sentence = sentence.replace(".", " ").lower()
    words = sentence.split(" ")

    indices = [i for i, x in enumerate(words) if x == target]
    if anaphor:
        target_index = indices[-1]
    else:
        target_index = indices[0]

    
    return target_index + 1
    # return words.index(target) + 1 # Adjust for [CLS]

def get_wordpiece_embeddings(og_index, tt, token_embeddings):
    """Assumes OOV word is first OOV word in sentence, so fine for these stims, but not general-purpose."""
    still_oov = True
    index = og_index + 1
    while still_oov:
        token = tt[index]
        if "##" not in token:
            break
        index += 1
    # Final set of embeddings
    emb = token_embeddings[og_index:index]
    # return mean across each token (should be (NUM_LAYER, NUM_UNITS) matrix)
    return emb.mean(dim = 0)


### Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

### Load models
BERT_MODEL = ['bert-base-uncased', 'bert-large-uncased']
model = BertModel.from_pretrained(BERT_MODEL[0],
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
model.eval()





### Code for iterating through stims
import itertools

import pandas as pd
import seaborn as sns

from scipy.spatial.distance import cosine
from tqdm import tqdm



### PATHS
STIMULI_PATH = "data/raw/similarity.csv"
SAVE_PATH = "data/processed/distances.csv"

### Read in stims
df_stims = pd.read_csv(STIMULI_PATH)
df_stims.head(5)

### Alter columns
df_stims['sentence'] = df_stims['Zeugmatic Similarity Norming Sentence']
df_stims['word'] = df_stims['CW'].str.lower()
df_stims['string'] = df_stims['String'].str.lower()

### Create new sentence that replaces anaphora

def replace_anaphora(row):
    sentence = row['sentence']
    target_word = row['string']
    anaphora = row['Anaphora']
    words = sentence.replace(".", "").split()
    revised = sentence.replace(anaphora, target_word)
    return revised





comparisons = []
not_identified = 0


with tqdm(total=len(df_stims)) as progress_bar:
    for index, row in df_stims.iterrows():

        target_word = row['string']
        sentence = row['sentence'].lower()
        ## Hand-coded
        anaphora = row['Anaphora']

        ## Remove these from stims
        if sentence[0] == "*":
            print(sentence)
            continue

        # Get embeddings
        tt, token_embeddings = get_embeddings_for_sentence(sentence)

        ## For now, ignore items where anaphora has multiple words
        if " " in anaphora:
            print(sentence)
            print(anaphora)
            continue

        ## For now, ignore items where anaphora can't be identified
        if anaphora not in tt:
            not_identified += 1
        #     continue

        if target_word not in tt:
            not_identified += 1
            

        # Embedding for target word in sentence 
        c_emb1 = get_embeddings_for_word(target_word, tt, token_embeddings, sentence)
        # Get embedding for anaphora
        ### TODO: Double-check this
        c_emb2 = get_embeddings_for_word(anaphora, tt, token_embeddings, sentence, anaphor=True)
        
        new_dict = {
            'string': target_word,
            'word': row['word'],
            'Similarity Norming Category': row['Similarity Norming Category']
            }

        ## Calculate distance for each layer
        for layer in range(1, 13):
            c1 = c_emb1[layer]
            c2 = c_emb2[layer]

            key = 'distance_bert_large_hf_layer_{i}'.format(i = str(layer))
            new_dict.update({key: cosine(c1, c2)})


        ### Now also do for revised sentence
        ### TODO: Need a way to get embeddings for each instance of target word
        """
        revised_sentence = row['sentence_revised']
        # Get embeddings
        tt, token_embeddings = get_embeddings_for_sentence(revised_sentence)
        # Embedding for target word in sentence 
        c_emb1 = get_embeddings_for_word(target_word, tt, token_embeddings, revised_sentence)
        # Get embedding for anaphora
        ### TODO: Double-check this
        c_emb2 = get_embeddings_for_word(anaphora, tt, token_embeddings, target_word)
        """


        comparisons.append(new_dict)


        
        progress_bar.update(1)


            
## Create dataframe
df_distances = pd.DataFrame(comparisons)
print(len(df_distances))
print(not_identified)

## Save data
df_distances.to_csv(SAVE_PATH)

