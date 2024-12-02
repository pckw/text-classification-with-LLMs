# Using Large Language Models for Classification of News Articles
This repo demonstrates how to use different LLMs to classify news articles and compare the result to a traditional XGBoost classifier. The details can be found in this [artice](https://medium.com/@p-wagner/using-llms-for-text-classification-17e53b3a7ead).

## Getting started

Copy config/config_template.yaml into config/config.yaml and set the variables with your own values before running the samples.
Add your api key to .env `APIKEY=...`

### Using a virtual environment

Install Python 3.10 and create a virtual environment named "venv" using
`python -m venv venv`
and install the required modules by first starting the virtual environment using
`venv\scripts\activate`
and then
`pip install -r requirements.txt`.

## Prepare the data and run the models
* Create a folder data/news/csv
* Download the [dataset](https://www.kaggle.com/datasets/banuprakashv/news-articles-classification-dataset-for-nlp-and-ml)
* Convert the csv data to a train.pkl and test.pkl using Scripts/prepare_news_data.py
* The jupyter-notebook main.ipynb provides an example on how to run the code.



