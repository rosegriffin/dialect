
# Speech-based American Dialect Identification

An ongoing machine learning project exploring and refining models for classifying American English regional dialects using acoustic speech features.

## Overview

Dialect identification is the process of determining the specific dialect or regional variety of a language that a speaker belongs to. An understanding of this is important as it improves the functionality of speech recognition systems, ensuring fairness across non-standard varieties as well as overall robustness. 

American regional dialects exhibit a variety of distinguishing acoustic patterns, but are not cleanly separable due to overlap, making classification between some pairs challenging. While these features can be observed objectively through phonetic analysis, this process is time consuming and difficult to scale. As such, I seek to automate dialect identification using machine learning.

In this project, I implement and compare a variety of models and approaches to identify American regional dialects acoustically using speech samples from the TIMIT dataset, with the goal of iteratively improving performance. The models take in speech clips (from TIMIT’s SA1/SA2 sentences) as input and output predicted dialect labels. So far, I have experimented with MFCCs and Wav2Vec embeddings as features, evaluating performance using balanced accuracy to account for class imbalance. As an early observation, some dialects are generally more difficult to classify than others, and that some models are better at identifying certain dialects than others.

## Dataset

TIMIT is a read speech corpus containing speech samples from speakers across the US. Each speaker is conveniently labeled as belonging to one of the following dialect regions:

| Region # | Region name              | # Speakers |
|----------|--------------------------|------------|
| DR1      | New England              | 49         |
| DR2      | Northern                 | 102        |
| DR3      | North Midland            | 102        |
| DR4      | South Midland            | 100        |
| DR5      | Southern                 | 98         |
| DR6      | New York City            | 46         |
| DR7      | Western                  | 100        |
| DR8      | Army Brat                | 33         |

For training and evaluation, only SA1 and SA2 sentences are used since they are spoken by all speakers in the dataset. I found that including the full dataset introduced too much variability, which degraded model performance.

- **SA1:** She had your dark suit in greasy wash water all year.
- **SA2:** Don't ask me to carry an oily rag like that. 

For more information, see the [LDC catalog page.](https://catalog.ldc.upenn.edu/LDC93S1)


## Getting started

### 1. Install Dependencies

Required libraries are listed in `environment.yml`. To create a conda environment with all required dependencies, run:

```
conda env create -f environment.yml
```

And to activate:
```
conda activate dialect
```

### 2. Prepare the Dataset

1. Download the TIMIT dataset.
2. Place the dataset in `data/raw`. The directory structure from root should follow TIMIT’s original layout, for example:  
`data/raw/TIMIT/data/TRAIN/DR1/FCJF0/SA1.WAV`
4. Run `split_samples.py` to split samples into words:  
```python -m amer_dialect_id.data.split_samples```

This will completely generate the `processed` folder, which will be used by the models.


### 3. Train and Interact with a Model

Train and evaluate a model:
```
python -m amer_dialect_id.main
```

Get predictions for samples:
```
python -m amer_dialect_id.utils.predict -m <model name> -s <paths to samples>
```

For example:
```
python -m amer_dialect_id.utils.predict -m wav2vec_lr -s SA1.WAV SA2.WAV
```
