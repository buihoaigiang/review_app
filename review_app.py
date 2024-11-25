import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
--%matplotlib inline
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud,STOPWORDS
from google_play_scraper import Sort, reviews
from app_store_scraper import AppStore
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
from typing import Union
import re
from functools import lru_cache
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import nltk
# from nltk import ngramspyth
from nltk.util import ngrams
import streamlit

def app_store_scraper(country, app_name, app_id, how_many):
    ly = AppStore(country=country, app_name=app_name, app_id = app_id)
    ly.review(how_many=how_many)
    columns = ['title', 'userName', 'review', 'rating', 'date']
    df = pd.DataFrame(ly.reviews, columns=columns)
    # remane columns
    columns = ['reviewId', 'userName', 'content', 'score', 'at']
    df.columns = columns
    return df

def googleplay_store_scraper(country, app_id, how_many):
    result, continuation_token = reviews(
                                        app_id = app_id,
                                        lang='en', 
                                        country=country, 
                                        sort=Sort.NEWEST, 
                                        count=how_many, 
                                        )
    result, _ = reviews(
        'com.fantome.penguinisle',
        continuation_token=continuation_token # defaults to None(load from the beginning)
    )
    columns = ['reviewId', 'userName', 'content', 'score', 'at']
    df = pd.DataFrame(result, columns=columns)
    return df


def save_df(store):
    if store == 'android':
        df_final = googleplay_store_scraper(country, app_id, how_many)
    else:
        df_final = app_store_scraper(country, app_name, app_id, how_many)
    
    df_final['event_date'] = pd.to_datetime(df_final['at'])
    df_final['event_date'] = df_final['event_date'].dt.normalize()
    return df_final

def viz():
    df = save_df(store)
    #Calculate reviews per dims
    review_per_day = df.groupby(['event_date'])['reviewId'].count().reset_index()
    cal_review_by_star = df.groupby(["event_date", "score"]).reviewId.count().reset_index()

    #plot
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), sharex = True, sharey = True)
    # Subplot 1: Total Reviews per Day (Seaborn)
    sns.lineplot(data=review_per_day, x="event_date", y="reviewId", marker='o', markersize=8, ax=axes[0])
    axes[0].set_xlabel("Event date", fontsize=12)
    axes[0].set_ylabel("Total review overtime", fontsize=12)
    axes[0].set_title("Total review overtime", fontsize=14)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)


    sns.lineplot(data=cal_review_by_star, x='event_date', y='reviewId', hue='score',palette='RdBu', marker='o', markersize=8, ax=axes[1])
    axes[1].set_ylabel("Total review overtime by score", fontsize=12)
    axes[1].set_title("Total review overtime by score", fontsize=14)
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel("Event date", fontsize=12)

    plt.tight_layout()
    plt.show()

# Initialize constants
STOP_WORDS = set(stopwords.words('english')) - {'no', 'not', 'nor', 'none', 'never', 'neither', 'barely', 'hardly', 'scarcely', 'rarely'}
WORD_PATTERN = re.compile(r'\w+')
NEGATION_WORDS = {'no', 'not', 'none', 'never', 'neither', 'barely', 'hardly', 'scarcely', 'rarely'}
lemma = WordNetLemmatizer().lemmatize

def clean_word(word: Union[str, float, int]) -> str:
    """
    Clean a single word by removing non-ASCII characters and punctuation,
    then lemmatize it while preserving negative words.
    """
    try:
        word = str(word).lower()
        
        # Preserve negation words as is
        if word in NEGATION_WORDS:
            return word
            
        # Normal cleaning for other words
        normalized = unicodedata.normalize('NFKD', word)
        ascii_text = normalized.encode('ASCII', 'ignore').decode('ASCII')
        cleaned = ''.join(char for char in ascii_text if char not in punctuation)
        
        # Lemmatize if word is not empty
        return lemma(cleaned) if cleaned else ''
    except:
        return ''

def clean_text(text: Union[str, float, int]) -> str:
    """
    Clean text while preserving negative words and their context.
    """
    if pd.isna(text):
        return ''
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Extract words
    words = WORD_PATTERN.findall(text)
    
    # Clean words while preserving negations and important context
    cleaned_words = []
    for i, word in enumerate(words):
        # Keep negation words
        if word in NEGATION_WORDS:
            cleaned_words.append(word)
            continue
            
        # Skip stopwords unless they follow a negation
        if word in STOP_WORDS:
            if i > 0 and words[i-1] in NEGATION_WORDS:
                cleaned_words.append(clean_word(word))
            continue
            
        cleaned = clean_word(word)
        if cleaned and len(cleaned) >= 1:
            cleaned_words.append(cleaned)
    
    return ' '.join(cleaned_words)

def process_reviews(df: pd.DataFrame, column: str) -> str:
    """
    Process reviews in a DataFrame column and combine them into a single string.
    """
    df[column] = df[column].apply(clean_text)
    return ' '.join(
        text for text in df[column] 
        if isinstance(text, str) and text.strip()
    )

nltk.download('punkt') # cần tải xuống nếu chưa có

# ... (code của bạn để đọc dữ liệu) ...

# Hàm tạo bigrams
def get_bigrams(text):
    tokens = nltk.word_tokenize(text)
    return [' '.join(x) for x in ngrams(tokens,1)] # tạo bigram
store = 'android'
country = 'us'
app_name = 'com.fantome.penguinisle'
app_id = 'com.fantome.penguinisle'
how_many = 100

viz()
to_viz_word_cloud = save_df(store)
to_viz_word_cloud['clened_content'] = to_viz_word_cloud['content'].apply(clean_text)

plt.figure(figsize=(20, 15))
for score in to_viz_word_cloud['score'].unique():
    text = ' '.join(to_viz_word_cloud[to_viz_word_cloud['score'] == score]['clened_content'])
    bigrams = get_bigrams(text)
    bigram_counts = Counter(bigrams)

    # Thiết lập màu nền dựa trên điểm số
    background_color = '#FFF0F5' if score < 3 else '#f0f0f6'

    wordcloud = WordCloud(width=1000, height=500, background_color=background_color).generate_from_frequencies(bigram_counts)
    plt.subplot(3, 2, score)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Score: {score}')
    plt.axis("off")

plt.tight_layout()
plt.show()
