import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image

path = (r'C:\Users\mupsi\Desktop\crowdsourced_curl\full_dataset_topics_24jun2.csv')
def load_csv(path):
    df1 = pd.read_csv(path, dtype='string')
    df2 = df1[['Link', 'Text', 'Hairtype','lemmatized_txt',  'First_Topic', 'Second_Topic', 'Third_Topic', 'Fourth_Topic']]
    return df2


cust_stopwords = text.ENGLISH_STOP_WORDS.union(['2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c', 'just', 've', 'wa', 'don', 'using', 'really', 'routine', 'know',
                                         'ha', 'cg', 'amp', 'work', 'try', 'used', 'x200b', 'notext', 'make', 'low', 'year', 'love', 'think', 'help', 'type', 'need', 'cut', 'lot', 'week', 'maybe', 'sure'])
tfidf = TfidfVectorizer(stop_words=cust_stopwords) #no n-gram here, go back and fix this BEFORE THE LEMMATIZATION

stream_dataset = load_csv(path)

st.title("The Crowdsourced Curl")
st.subheader("Find the product that's right for your curls using thousands of posts from r/curlyhair")

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

    if len(top_docs) == 0:
        st.write("No results")
    else:
        top_output = df_dum.loc[top_docs]
        output = top_output[['Hairtype', 'Text','First_Topic', 'Second_Topic', 'Third_Topic', 'Fourth_Topic']]
        output = output.sort_values(by=['Hairtype'])
        st.write("Posts are sorted by hair type. Hover your mouse over the post to read the whole post. Scroll to see the four main topics covered in the post.")
        st.write(output)
        return(output)
    
searchtext = st.text_input("What product would you like to learn about?")

if searchtext:
    search_results = curl_query(searchtext, stream_dataset, tfidf)
    
    image = Image.open(r'C:\Users\mupsi\Desktop\crowdsourced_curl\hair_types.png')
    st.sidebar.subheader("Don't know your hair type? Check here: ")
    st.sidebar.image(image, width=450)

    topics = stream_dataset['Third_Topic'].unique()
    topic_choice = st.multiselect("Filter by topic: ", topics)

               
    if st.button("Submit topic filter"):
            search_results= search_results.loc[search_results['First_Topic'].isin(topic_choice)\
            | search_results['Second_Topic'].isin(topic_choice) |search_results['Third_Topic'].isin(topic_choice)\
                                                      | search_results['Fourth_Topic'].isin(topic_choice)]
            st.write(search_results)
    else:
        st.write(" ")
else:
    st.write(" ")


 ########## to filter by hairtype: not necessary given hairtype column: ##########   
    
#    image = Image.open(r'C:\Users\mupsi\Desktop\crowdsourced_curl\hair_types.png')
#    st.subheader("Check your hair type and filter your results: ")
#    st.image(image, width=450)#

#    hairtypes = search_results['Hairtype'].unique()
#    hairtype_choice = st.multiselect('',hairtypes)
    
#    if st.button("Submit hair type"):
#        search_results_hair = search_results[search_results['Hairtype'].isin(hairtype_choice)]
 #       st.write(search_results_hair)
        
#        topics = stream_dataset['Third_Topic'].unique()
#        topic_choice = st.multiselect("Filter by topic: ", topics)
#                
#        if st.button("Submit topic filter"):
#            search_results_hair_topic = search_results_hair.loc[search_results['First_Topic'].isin(topic_choice)| search_results_hair['Second_Topic'].isin(topic_choice) |search_results_hair['Third_Topic'].isin(topic_choice)| search_results_hair['Fourth_Topic'].isin(topic_choice)]
#            st.write(search_results_hair_topic, astype('object'))
#else:
#    st.write('')
