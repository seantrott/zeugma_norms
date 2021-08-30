# Zeugma Norms

Code to reproduce the analyses described in:

> Offline dominance and zeugmatic similarity normings of variably ambiguous words assessed against a neural language model (BERT). (Submitted). DeLong, K., Trott, S., Kutas, M.

## To run BERT

Code for running BERT and obtaining `surprisal` and `cosine distance` estimates is included in `src/modeling/surprisal.py` and `src/modeling/distances.py`, respectively.

They can be run using Python:

```
python src/modeling/distances.py
python src/modeling/surprisal.py
```

Running this code will produce two data files:

- `data/processed/distances.csv`  
- `data/processed/surprisals.csv`

Note that the code was written and evaluated using Python version **3.7.1**. 

Additionally, the code has a number of dependencies:

- `torch`  
- `transformers`
- `scipy`
- `nltk`
- `pandas`
- `numpy`

These can be installed from command line using `pip`:

```
pip install torch
```

Alternatively, some pakcages (`pandas`, `scipy`, etc.) are available through [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).

## To run and view analyses

The analysis of BERT `surprisal` (and `cosine distance`) is included in `src/analysis/norming_analysis.Rmd`. This file can be "knit" into an `.html` file using RStudio; we've also already included the compiled/knit `.html` file for viewing.


## Contact

If you have any questions about the data (or supplementary analyses), please contact Sean Trott: sttrott at ucsd dot edu.


