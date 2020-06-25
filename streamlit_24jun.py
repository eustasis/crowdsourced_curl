#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel


# In[4]:


path = (r'C:\Users\mupsi\Desktop\crowdsourced_curl\full_dataset_topics_24jun2.csv')
def load_csv(path=r'C:\Users\mupsi\Desktop\crowdsourced_curl\full_dataset_topics_24jun2.csv'):
    df1 = pd.read_csv(path, dtype='string')
    df2 = df1[['Link', 'Text', 'Hairtype','lemmatized_txt',  'First_Topic', 'Second_Topic', 'Third_Topic', 'Fourth_Topic']]
    return df2


# In[7]:


cust_stopwords = text.ENGLISH_STOP_WORDS.union(['2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c', 'just', 've', 'wa', 'don', 'using', 'really', 'routine', 'know',
                                         'ha', 'cg', 'amp', 'work', 'try', 'used', 'x200b', 'notext', 'make', 'low', 'year', 'love', 'think', 'help', 'type', 'need', 'cut', 'lot', 'week', 'maybe', 'sure'])
tfidf = TfidfVectorizer(stop_words=cust_stopwords) #no n-gram here, go back and fix this BEFORE THE LEMMATIZATION


# In[5]:


stream_dataset = load_csv(path)

st.title("The Crowdsourced Curl")
st.subheader("Find the product that's right for your curls using thousands of posts from r/curlyhair")



#output_choice = output[output['hairtype'].isin(hairtype_choice)]

def curl_query(product, stream_dataset, tfidf):
    """
    Return most similar based on cosine similarity
    
    """
    df_dum = pd.DataFrame([product], columns=['lemmatized_txt'])
    df_dum = pd.concat([stream_dataset, df_dum])
    query_vectors = tfidf.fit_transform(df_dum['lemmatized_txt'])
    cosine_similarities=linear_kernel(query_vectors[-1], query_vectors).flatten()
    related_docs = np.where((cosine_similarities>0.3) & (cosine_similarities<1))
    
    for i in related_docs:
        top_docs = []
        top_docs.extend(i)
        
    if len(top_docs)>1:
        top_output = df_dum.loc[top_docs]
        output = top_output[['Hairtype', 'Text','First_Topic', 'Second_Topic', 'Third_Topic', 'Fourth_Topic','Link']]
        output = output.sort_values(by=['Hairtype'])
        st.write("Hover your mouse over the post to read the whole thing. \r Scroll to see what the main topics covered in the post are. \r Click the permalink to go to the original post.")
        st.write(output)
    else:
        st.write("No results found; try a different search!")

searchtext = st.text_input("What product would you like to learn about?")

if searchtext:
    search_results = curl_query(searchtext, stream_dataset, tfidf)
    search_results

    topics = stream_dataset['Third_Topic'].unique()
    topic_choice = st.multiselect("Filter by topic",topics)
    if st.button("Submit"):
        st.write('ok')
    else:
        st.write('You can select more than one!')

        
    
