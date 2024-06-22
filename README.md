# MLOps Fake News Classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps Project of Tobias Brock, Anne Gritto und Stefanie Schwarz.

This project is part of the fullfilment for the lecture "ML Operations" from LMU and covers the whole deployment lifecycle of a deep learning project. Here we deploy a model for fake news classification.

## Project Description: Fake News Classifier

### Overall Goal of the Project

The Fake News Classifier project aims to equip students in the ML Operations (MLOps) lecture with hands-on experience in deployment and development techniques. The task of fake news classification is chosen since it offers a straightforward implementation using transformer-based libraries. The goal is to create a robust machine learning model that accurately distinguishes between real and fake news articles.

### Framework and Integration

For the classification task we use PyTorch in combination with Hugging Face's Transformers library. PyTorch is a popular deep learning framework known for its flexibility and ease of use. The Transformers library provides pre-trained models and APIs for implementing transformer architectures like BERT (Bidirectional Encoder Representations from Transformers) and SBERT (Sentence-BERT), which are effective for Natural Language Processing (NLP) tasks.

To integrate these frameworks, a conda environment is used to manage dependencies and ensure compatibility. Necessary packages like PyTorch, Transformers, and others are installed using pip. The project structure includes scripts for data preprocessing, model training, and evaluation.

### Data

For the project, the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data) from Kaggle is used. This dataset contains 72,134 news articles, with 35,028 labeled as real and 37,106 as fake. It combines four popular news datasets (Kaggle, McIntire, Reuters, BuzzFeed Political) to prevent over-fitting and provide a larger corpus for training.

The dataset includes:
1. **Serial Number**: A unique identifier for each entry.
2. **Title**: The headline of the news article.
3. **Text**: The content of the news article.
4. **Label**: A binary label indicating whether the news is fake (0) or real (1).

### Models

We focus on transformer-based models, specifically BERT and SBERT:
1. **DistilBERT**: Distilled version of BERT without fine-tuning on the training data should serve as baseline model.

2. **BERT**: BERT is pre-trained on a large corpus and we will further fine-tune the model for our specific tasks of fake news classification. Its contextual embeddings will help improve the classifier's accuracy in distinguishing real and fake news.

3. **SBERT**: SBERT is a variant of BERT designed to produce semantically meaningful sentence embeddings, making it suitable for understanding sentence-level semantics in news articles.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for fakenews
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── fakenews                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes fakenews a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------
