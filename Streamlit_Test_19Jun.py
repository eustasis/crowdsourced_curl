import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel

#import dataset, choose only relevant columns
df1 = pd.read_csv(r'reddit_dataset', dtype='string')
df2 = df1[['author', 'id', 'permalink', 'created_utc', 'text_body', 'parent_id', 'hairtype', 'num_comments','num_crossposts', 'selftext', 'title', 'lemmatized_txt']]


# instantiate vectorizer with custom stopwords
cust_stopwords = text.ENGLISH_STOP_WORDS.union(['2a','2b','2c','3a','3b','3c','4a','4b','4c'])
tfidf = TfidfVectorizer(stop_words=cust_stopwords, tokenizer=None)

st.title("The Crowdsourced Curl")
st.subheader("Find the product that's right for your curls using thousands of posts from r/curlyhair")

# function to re-vectorize search term along with each reddit post, return search results as a df with the post (text & title = 'text_body'), associated hairtype, and permalink to the original post
def curl_query(str):   
	df = df2
	df.append=['author', 'id', 'permalink', 'created_utc', 'text_body', 'parent_id','hairtype', 'num_comments','num_crossposts', 'selftext', 'title', str]
	query_vectors=tfidf.fit_transform(df['lemmatized_txt'])
	cosine_similarities=linear_kernel(query_vectors[0], query_vectors).flatten()
	related_docs = np.where((cosine_similarities>0.2) & (cosine_similarities<1))

	search_results = []
	for i in related_docs:
		global output
		result = df.iloc[i].copy()
		search_results.append(result)
		search_results = pd.concat(search_results)
		output = search_results[['permalink','text_body','hairtype']]
		output = output.drop_duplicates(subset='permalink', keep='first')
		return output
		#st.write(output.astype('object'))

# searchbox for user input & hairtype filter
searchtext = st.text_input("What product would you like to learn about?")
hairtypes = df2['hairtype'].unique()

if st.button("Submit"):
	curl_query(searchtext)
	st.write(output.astype('object'))

	hairtype_choice = st.multiselect('Choose your hairtype:', hairtypes)
	if hairtype_choice:
		output_choice = output[output['hairtype']==hairtype_choice]
		st.write(output_choice.astype('object'))
		

