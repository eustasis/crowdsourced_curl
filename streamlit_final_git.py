import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image
~
path = (r'~final_dataset_for_streamlit.csv')
def load_csv(path):
    df1 = pd.read_csv(path, dtype='string')
    df2 = df1[['Link', 'Raw_Text', 'Hairtype','First_Topic_Name', 'Second_Topic_Name', 'Third_Topic_Name', 'Fourth_Topic_Name']]
    return df2


tfidf = TfidfVectorizer(stop_words=cust_stopwords) 

stream_dataset = load_csv(path)

st.title("The Crowdsourced Curl")
st.subheader("Find the product that's right for your curls using thousands of posts from r/curlyhair")

def curl_query(product, stream_dataset, tfidf):
    """
    Return most similar based on cosine similarity
    
    """
    df_dum = pd.DataFrame([product], columns=['Raw_Text'])
    df_dum = pd.concat([stream_dataset, df_dum])
    query_vectors = tfidf.fit_transform(df_dum['Raw_Text'])
    cosine_similarities=linear_kernel(query_vectors[-1], query_vectors).flatten()
    related_docs = np.where((cosine_similarities>0.3) & (cosine_similarities<1))
    
    for i in related_docs:
        top_docs = []
        top_docs.extend(i)

    if len(top_docs) == 0:
        st.write("No results")
    else:
        top_output = df_dum.loc[top_docs]
        output = top_output[['Hairtype', 'Raw_Text','First_Topic_Name', 'Second_Topic_Name', 'Third_Topic_Name', 'Fourth_Topic_Name']]
        output = output.sort_values(by=['Hairtype'])
        st.write("Posts are sorted by hair type. Hover your mouse over the post to read the whole post. Scroll to see the four main topics covered in the post.")
        st.write(output)
        return(output)
    
searchtext = st.text_input("What product would you like to learn about?")

if searchtext:
    search_results = curl_query(searchtext, stream_dataset, tfidf)
    
    image = Image.open(r'~\hair_types.png')
    st.sidebar.subheader("Don't know your hair type? Check here: ")
    st.sidebar.image(image, width=450)

    topics = stream_dataset['Third_Topic_Name'].unique()
    topic_choice = st.multiselect("Filter by topic: ", topics)

               
    if st.button("Submit topic filter"):
            search_results= search_results.loc[search_results['First_Topic_Name'].isin(topic_choice)\
            | search_results['Second_Topic_Name'].isin(topic_choice) |search_results['Third_Topic_Name'].isin(topic_choice)\
                                                      | search_results['Fourth_Topic_Name'].isin(topic_choice)]
            st.write(search_results)
    else:
        st.write(" ")
else:
    st.write(" ")
