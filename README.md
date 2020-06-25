# The Crowdsourced Curl
 Create a TF-IDF search function to search r/curlyhair posts for product information based on your hairtype, and assign topics using Latent Dirichlet Allocation. This project was developed using data scraped from PushShift. The dataset:
 - was scraped by searching for posts containing each hairtype (data from 8-Jun-2018 to 8-Jun-2020)
 - each post been labeled with an associated hairtype; posts with multiple hairtypes are duplicated
 - initial posts in Reddit ('submissions) have both a title ('title' column) and a body ('body' column), while comments only have a body ('selftext' column); here titles and bodies have been combined per post, and added to a new column, along with selftexts from comments, into the 'Text' column.
 
 Deployed using Streamlit and AWS. 
