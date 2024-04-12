# YouTube Comments Sentiment Analyzer
This Python script analyzes sentiment in YouTube video comments using machine learning techniques. It retrieves comments from a specified YouTube video, preprocesses the text data, applies sentiment analysis using various classifiers, and generates classification reports.

## Installation
Clone the repository to your local machine:

### git clone https://github.com/aryapg/Youtube-Comments-Sentiment-Analyzer.git

## Install the required Python packages using pip:

### pip install -r requirements.txt

## Usage

1) Obtain a YouTube Data API key from the Google Cloud Console.
2) Add your YouTube Data API key to the script:
   . Open the youtube_comments_sentiment.py file in a text editor.
   . Locate the api_key variable in the script.
   . Replace "YOUR_API_KEY" with your actual API key obtained from the Google Cloud Console.
    ### api_key = "YOUR_API_KEY"
   
3) Run the script and enter the YouTube video ID when prompted.
    ### python youtube_comments_sentiment_analyser.py

## Requirements
. Python 3.6 or higher
. pandas
. google-api-python-client
. scikit-learn
. textblob
. nltk
. emoji
. gensim
. matplotlib
. numpy (if explicitly used in your code)

## File Descriptions
. youtube_comments_sentiment.ipynb and .py: Main Python script for sentiment analysis on YouTube comments.
. requirements.txt: List of required Python packages and their versions.

## Acknowledgments
This project utilizes the YouTube Data API v3 and various machine learning libraries for sentiment analysis.

