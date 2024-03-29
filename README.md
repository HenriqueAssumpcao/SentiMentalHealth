# SentiMentalHealth
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

PyTorch implementation of our GRU-based RNN for predicting user emotional tone in online mental disorder communities.

<img src='assets/gru.png' align="center" width=900 />

## Introduction

This repository provides the original implementation of the "Predicting User Emotional Tone in Mental Disorder Online Communities". The paper was published in the [FGCS Special Issue on Sentiment Analysis Systems](https://www.sciencedirect.com/science/article/pii/S0167739X21002764?dgcid=author).
The chosen model for extracting the sentiment features from reddit posts and comments was the [VADER Sentiment Analysis Tool](https://github.com/cjhutto/vaderSentiment).

## About the Dataset

The paper used a dataset comprised of four different subreddits related to mental health issues, namely [r/Anxiety](https://www.reddit.com/r/Anxiety/), [r/bipolar](https://www.reddit.com/r/bipolar/), [r/depression](https://www.reddit.com/r/depression/) and [r/SuicideWatch](https://www.reddit.com/r/SuicideWatch/). The following table provides a detailed statistical description of each.

<img src='assets/dataset.png' align="center" width=750 />

## Contents and Usage

This repository contains four python notebooks, each accomplishing a specific part of the proposed pipeline. We highlight that all the notebooks are already configured for enabling users to reproduce the paper's results. 

* The [REDDIT_SCRAPPER](REDDIT_SCRAPPER.ipynb) notebook downloads the data from a given subreddit using reddit's API. 
* The [PRE_PROCESSING](PRE_PROCESSING.ipynb) notebook transforms the data into a single dataframe containing all the information needed for training(VADER's scores, relevant statistics about the threads, etc.)
* The [PAPER_RESULTS](PAPER_RESULTS.ipynb) notebook performs the actual training of the model and the analysis of the results.
* The [REVIEW_EXPERIEMENTS](REVIEW_EXPERIMENTS.ipynb) notebook is an extra part of the pipeline, that performs confidence interval experiments and other relevant but not essential studies for the model.

The [Source](src) folder contains the .py files with useful functions and, most importantly, the Pytorch implementation of the model itself. 

### First steps
For correct usage of the code in this repository, follow these two simple steps:
1. The following is a description of the correct file structure for the repository files.
```
SentiMentalHealth/
└── src/
└── data/
└── results/
└── model/
└── PAPER_RESULTS.ipynb
└── PRE_PROCESSING.ipynb
└── REDDIT_SCRAPER.ipynb
└── REVIEW_EXPERIMENTS.ipynb
└── scraper.py
└── requirements.txt
```
   *All* of the notebooks already check if the root directory follows the correct structure, and otherwise create the required files, so if you are using the code according to this guide your file structure will be correct.
   
2. Install all of the requirements:
```
pip install -r requirements.txt
```
### Reproducing the Paper's results
The main goal of this repository is reproducibility, and so we succinctly describe the pipeline one should follow to reproduce our results.
1. Run the [REDDIT_SCRAPPER](REDDIT_SCRAPPER.ipynb) notebook. We recommend running this locally, since Google Colab doesn't handle asynchronous functions all that well. This can take a while since subreddits like r/depression contain a significant amount of threads.

    *Alternatively*, you can download the compressed dataset files from the authors' OneCloud drive [here](https://ufmgbr-my.sharepoint.com/:u:/g/personal/henriquesas2020_ufmg_br/EfMxFHP06HVMtxgii7xlSfYBxPkXr9eIXrmnIQLkp-3I-A?e=LphyOv).

    Do NOT forget to maintain the directory structure of the zip file, i.e., keep all the .pkl files in the 'data' directory.


2. Run the [PRE_PROCESSING](PRE_PROCESSING.ipynb) notebook. We recommend running this on colab, since it requires TPU support. This will also take a while since obtaining the VADER sentiment scores is a costly task.

    *Alternatively*, you can skip Step 1 and 2 entirely and download the preprocessed data (parquet format) from the authors' OneCloud drive [here](https://ufmgbr-my.sharepoint.com/:u:/g/personal/henriquesas2020_ufmg_br/EQcbKtFvQOdKnlGUVkHjiS0BgMpvQu-3cC_q1FGOIgl4wA?e=iM4wx1).

    Do NOT forget to maintain the directory structure of the zip file, i.e., keep all the .pkl files in the 'data' directory, and to configure the [PAPER_RESULTS](PAPER_RESULTS.ipynb) notebook accordingly, i.e., set ```EXTENSION = ".parquet"```.

There are two options for the final step of the pipeline.
#### Option 1: Training the model from Scratch
3. Run the [PAPER_RESULTS](PAPER_RESULTS.ipynb) notebook. Unfortunately, since we first obtained the results, Google Colab has reduced the amount of available RAM for free users, and so training the model for all 4 subreddits is not currently possible in the free version of Colab. For this reason, we recommend running this notebook locally, on a machine with GPU support and more than 14GB of RAM. 
#### Option 2: Loading pre-trained parameters 
3. Download the pre-trained parameters pickle file from the authors' OneCloud drive [here](https://ufmgbr-my.sharepoint.com/:u:/g/personal/henriquesas2020_ufmg_br/EWGOcs3FyxJGqLnmU8lKwBYBPdK1tD1BC1RDzwpxqu4P1g?e=y6MedU) and then run the [PAPER_RESULTS](PAPER_RESULTS.ipynb) notebook in testing mode, i.e., set the variables ```TEST = True``` and ```TRAIN = False```.

Do NOT forget to maintain the directory structure of the zip file, i.e., keep the .pkl file in the 'results' directory.

This file contains the RNN state_dict, and then proceed only to test the model using the downloaded data. We still recommend using a machine with GPU support and more than 14GB of RAM, since Colab may have issues loading all of the subreddits data.

After following these three steps, you will have access to the trained model used in the paper. The [PAPER_RESULTS](PAPER_RESULTS.ipynb) notebook automatically generates the results table and performs case studies for specific threads.

## Future Features \& Questions
We plan to release a command-line version of the model in the future for easier and more general use, but currently the aforementioned pipeline has to be followed in order to use the model. If you have any suggestions or questions about the paper and/or about the repository itself, feel free to contact us via [email](mailto:henriquesoares@dcc.ufmg.br?subject=FGCS%3A%20%22Predicting%20User%20Emotional%20Tone%20in%20Mental%20Disorder%20Online%20Communities%22). 

## License \& Disclaimer
This is research code, expect that it can change and any fitness for a particular purpose is disclaimed.

This software is under GNU General Public License Version 3 ([GPLv3](LICENSE)).

## Authors
Bárbara Silveira, Henrique S. A. Silva, Fabricio Murai, Ana Paula C. da Silva

### Cite the Paper (BibTeX)
```
@article{SILVEIRA2021641,
title = {Predicting user emotional tone in mental disorder online communities},
journal = {Future Generation Computer Systems},
volume = {125},
pages = {641-651},
year = {2021},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2021.07.014},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X21002764},
author = {Bárbara Silveira and Henrique S. Silva and Fabricio Murai and Ana Paula C. {da Silva}},
keywords = {Emotional tone, Sentiment analysis, Mental health disorders, Reddit, Online communities, Emotional tone prediction},
abstract = {In recent years, Online Social Networks have become an important medium for people who suffer from mental disorders to share moments of hardship, and receive emotional and informational support. In this work, we analyze how discussions in Reddit communities related to mental disorders can help improve the health conditions of their users. Using the emotional tone of users’ writing as a proxy for emotional state, we uncover relationships between user interactions and state changes. First, we observe that authors of negative posts often write rosier comments after engaging in discussions, indicating that users’ emotional state can improve due to social support. Second, we build models based on SOTA text embedding techniques and RNNs to predict shifts in emotional tone. This differs from most of related work, which focuses primarily on detecting mental disorders from user activity. We demonstrate the feasibility of accurately predicting the users’ reactions to the interactions experienced in these platforms, and present some examples which illustrate that the models are correctly capturing the effects of comments on the author’s emotional tone. Our models hold promising implications for interventions to provide support for people struggling with mental illnesses.}
}
```

<img align="left" width="auto" height="75" src="./assets/ufmg.png">
