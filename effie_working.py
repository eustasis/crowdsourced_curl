#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel

path = 'full_reddit_dataset_ec2_18jun.csv'
def load_csv(path='reddit_for_daisy.csv'):
    df1 = pd.read_csv(path, dtype='string')
    # df1 = pd.read_csv()
    df2 = df1[['author', 'id', 'permalink', 'created_utc', 'text_body', 'parent_id',
        'hairtype', 'num_comments', 'num_crossposts', 'selftext', 'title', 'lemmatized_txt']]
    return df2


st.title("The Crowdsourced Curl")
st.subheader("Find the product that's right for your curls using thousands of posts from r/curlyhair")
df1 = pd.read_csv(path, dtype='string')
# df1 = pd.read_csv()
df2 = df1[['author', 'id', 'permalink', 'created_utc', 'text_body', 'parent_id',
    'hairtype', 'num_comments', 'num_crossposts', 'selftext', 'title', 'lemmatized_txt']]

def curl_query(product, df2, tfidf, topn=5):
    """

    Return top n most similar

    """
    df_dum = pd.DataFrame([product], columns=['lemmatized_txt'])
    df_dum = pd.concat([df2, df_dum])
    query_vectors = tfidf.fit_transform(df_dum['lemmatized_txt'])
    cosine_similarities=linear_kernel(query_vectors[-1], query_vectors).flatten()
#    related_docs = np.where((cosine_similarities>0.2) & (cosine_similarities<1))

    # so we can return top n without a harsh threshold
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != int(len(df_dum)-1)]

    top_n_output = df_dum.loc[related_docs_indices[:topn]]
    top_n_output = top_n_output[['permalink','text_body','hairtype']]
    output = top_n_output.drop_duplicates(subset='permalink', keep='first')
    return output


# df2 = load_csv()
cust_stopwords = text.ENGLISH_STOP_WORDS.union(['2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c', 'just', 've', 'wa', 'don', 'using', 'really', 'routine', 'know',
                                         'ha', 'cg', 'amp', 'work', 'try', 'used', 'notext', 'make', 'low', 'year', 'love', 'think', 'help', 'type', 'need', 'cut', 'lot', 'week', 'maybe', 'sure'])
tfidf = TfidfVectorizer(stop_words=cust_stopwords,
                  ngram_range=(1, 4), tokenizer=None)
searchtext = st.text_input("What product would you like to learn about?")
output = curl_query(searchtext, df2, tfidf)
st.write(output)

# filter by hairtype
hairtypes = df2['hairtype'].unique()
hairtype_choice = st.selectbox('Filter results by hairtype', hairtypes)
hairtype_choice = hairtype_choice.split()
output_choice = output[output['hairtype'].isin(hairtype_choice)]
if st.button("Submit"):
    st.write(output_choice)

