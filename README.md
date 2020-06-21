# crowdsourced_curl
 Create a TF-IDF search function to search r/curlyhair posts for product information based on your hairtype. This project was developed using data scraped from PushShift. The dataset here has:
 - already been scraped
 - been labeled with associated hairtypes for each post
 - had its text combined--initial posts in Reddit ('submissions) have both a title ('title' column) and a body ('body' column), while comments only have a body ('selftext' column); here titles and bodies have been combined per post, and added to a new column, along with selftexts from comments, into a new 'text_body' column.
 
 Deployed using Streamlit and AWS. 
