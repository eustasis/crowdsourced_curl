
# coding: utf-8

# In[1]:
# In[4]:


#df1 = pd.read_csv(r'C:\Users\mupsi\Desktop\crowdsourced_curl\full_reddit_dataset_ec2_18jun.csv', dtype='string')

#df2 = df1[['author', 'id', 'permalink', 'created_utc', 'text_body', 'parent_id', 'hairtype', 'num_comments','num_crossposts', 'selftext', 'title', 'lemmatized_txt']]

# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel


# In[4]:

@st.cache(allow_output_mutation=True)
def load_csv(path=r'C:\Users\mupsi\Desktop\reddit_for_daisy.csv'):
    df1 = pd.read_csv(path, dtype='string')
    # df1 = pd.read_csv()
    df2 = df1[['author', 'id', 'permalink', 'created_utc', 'text_body', 'parent_id',
        'hairtype', 'num_comments', 'num_crossposts', 'selftext', 'title', 'lemmatized_txt']]
    return df2


def setup_NLP_stuff():
  cust_stopwords = text.ENGLISH_STOP_WORDS.union(['2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c', 'just', 've', 'wa', 'don', 'using', 'really', 'routine', 'know',
                                                 'ha', 'cg', 'amp', 'work', 'try', 'used', 'notext', 'make', 'low', 'year', 'love', 'think', 'help', 'type', 'need', 'cut', 'lot', 'week', 'maybe', 'sure'])
  tfidf = TfidfVectorizer(stop_words=cust_stopwords,
                          ngram_range=(1, 4), tokenizer=None)
  return tfidf



def curl_query(product, df, tfidf):
    df.loc[0, 'lemmatized_txt'] = product
    query_vectors = tfidf.fit_transform(df['lemmatized_txt'])
    cosine_similarities = linear_kernel(
        query_vectors[0], query_vectors).flatten()
    related_docs = np.where((cosine_similarities > 0.23)
                            & (cosine_similarities < 1))

    search_results = []
    for i in related_docs:
        result = df.iloc[i].copy()
        search_results.append(result)
        search_results = pd.concat(search_results)
        output = search_results[['permalink', 'text_body', 'hairtype']]
        output = output.drop_duplicates(subset='permalink', keep='first')
        return output
        # st.write(output.astype('object'))


def main():
    df = load_csv()
    tfidf = setup_NLP_stuff()

    st.title("The Crowdsourced Curl")
    st.subheader(
        "Find the product that's right for your curls using thousands of posts from r/curlyhair")

  # def make_clickable(url, text):
  #        return f'<a target="_blank" href="{url}">{text}</a>

    searchtext = st.text_input("What product would you like to learn about?")
#    hairtypes = df['hairtype'].unique()
#    hairtype_choice = st.multiselect('Filter results by hairtype', hairtypes)

    if st.button("Submit"):
        output = curl_query(searchtext, df, tfidf)
        st.write(output.astype('object'))
#        if hairtype_choice:
#            output_choice = output[output['hairtype'].isin(hairtype_choice)]
#            st.write(output_choice.astype('object'))
#        elif len(hairtype_choice) == 0:
#            st.write(output)

if __name__ == '__main__':
      main()


#
