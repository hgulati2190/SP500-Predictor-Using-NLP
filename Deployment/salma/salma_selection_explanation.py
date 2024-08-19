import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime


# In[159]:


# Load the models and transformers
with open('./salma/salma_log_reg.pkl', 'rb') as file:
    log_reg = pickle.load(file)

with open('./salma/salma_xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

with open('./salma/salma_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('./salma/salma_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    


# In[190]:


# save_functions.py
import pickle
from salma_functions import clean_text, collect_headlines, fetch_spy_data

def save_functions(functions, filename):
    with open(filename, 'wb') as file:
        pickle.dump(functions, file)

functions_dict = {
    'clean_text': clean_text,
    'collect_headlines': collect_headlines,
    'fetch_spy_data': fetch_spy_data
}

save_functions(functions_dict, './salma/salma_functions.pkl')


# In[198]:


# load_functions.py
import pickle

def load_functions(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

functions_dict = load_functions('./salma/salma_functions.pkl')
print("Functions Loaded:", functions_dict)

clean_text_func = functions_dict['clean_text']
collect_headlines_func = functions_dict['collect_headlines']
fetch_spy_data_func = functions_dict['fetch_spy_data']


# In[214]:


# Streamlit app layout
st.title('Stock Movement Prediction')

# User inputs
# Convert to DataFrame
df_headlines = pd.DataFrame(collect_headlines_func())
# Apply clean_text function to the 'text' column
df_headlines['cleaned_text'] = df_headlines['text'].apply(clean_text)


# In[250]:


# Select the 'text' column
text_input = df_headlines['cleaned_text']


open_price_data = fetch_spy_data(datetime.datetime.now())

# Display the headlines
st.write("### News Headlines")
st.dataframe(df_headlines[['text']])

st.write("### Opening Price")
st.write(open_price_data[0]['Open'])

if st.button('Predict'):
    # Transform the input
    text_tfidf = vectorizer.transform(text_input)
    open_price_input = open_price_data[0]['Open']
        
        # Transform the opening price, reshape to 2D array
    open_scaled = scaler.transform(np.array([[open_price_input]]))
    
    # Make predictions
    text_prediction = log_reg.predict(text_tfidf)
    open_prediction = xgb_model.predict(open_scaled)
    
    # Combine predictions
    combined_prediction = (text_prediction[0] + open_prediction[0]) / 2
    final_prediction = int(np.round(combined_prediction))
    
    st.write(f'Predicted Movement: {"Up" if final_prediction == 1 else "Down"}')


# In[ ]:




