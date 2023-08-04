# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:48:42 2023

@author: ajith
"""

import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vect = TfidfVectorizer()


vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
  con = re.sub('[^a-zA-Z]', ' ', content)
  con = con.lower()
  con = con.split()
  con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
  con = ' '.join(con)
  return con

def threads_instagram_review(review):
  review = stemming(review)
  input_data = [review]
  vector_form1 = vector_form.transform(input_data)
  prediction = load_model.predict(vector_form1)
  return prediction

def main():
    
    
    st.set_page_config(page_title = "Threads, an Instagram app - User Reviews", page_icon='https://res.cloudinary.com/practicaldev/image/fetch/s--BzACho1h--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_800/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/xgt3pask0b39s915jpt7.jpg', layout="wide")
    
    def add_image_from_url():
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("https://marketinglabs.co.uk/wp-content/uploads/2023/07/Threads-app-from-Meta.jpeg");
                    background-attachment: fixed;
                    background-size: cover
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
    
    add_image_from_url()
    
    st.title('Threads, an Instagram app - User Reviews')
    st.write('***Created by Ajith Muraleedharan***')
    sentence = st.text_area('       ', 'User Review', height=100)
    rating_btt = st.button('**App Rating**')
    if rating_btt:
        rating_class = threads_instagram_review(sentence)
        print(rating_class)
        if rating_class == [1]:
            st.error('**Unacceptable**')
        elif rating_class == [2]:
            st.warning('**Needs Improvement**')
        elif rating_class == [3]:
            st.success('**Meets Expectations**')
        elif rating_class == [4]:
            st.success('**Exceeds Expectations**')
        else:
            st.success('**Outstanding**')
            
            
if __name__ == '__main__':
    main()
    
