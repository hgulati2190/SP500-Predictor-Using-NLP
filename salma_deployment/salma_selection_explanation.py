import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
import nltk

import os
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir(os.getcwd()))

if os.path.exists('salma_dll'):
    print("Files in salma_dll folder:", os.listdir('salma_dll'))
else:
    print("'salma_dll' folder not found in current directory")
print("Files in salma_dll folder:", os.listdir('salma_dll'))
base_dir = os.path.dirname(__file__)
log_reg_path = os.path.join(base_dir, 'salma_dll', 'salma_log_reg.pkl')
with open(log_reg_path, 'rb') as file:
    log_reg = pickle.load(file)

# Load the models and transformers
with open('salma_dll/salma_log_reg.pkl', 'rb') as file:
    log_reg = pickle.load(file)

with open('salma_dll/salma_xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

with open('salma_dll/salma_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('salma_dll/salma_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load functions
def load_functions(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
nltk.download('punkt_tab')
nltk.download('punkt', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')

# Add the path to nltk data
nltk.data.path.append('nltk_data')

functions_dict = load_functions('salma_functions.pkl')
clean_text_func = functions_dict['clean_text']
collect_headlines_func = functions_dict['collect_headlines']
fetch_spy_data_func = functions_dict['fetch_spy_data']

# Streamlit app layout
st.title('Stock Movement Prediction')

# User inputs
df_headlines = pd.DataFrame(collect_headlines_func())
df_headlines['cleaned_text'] = df_headlines['text'].apply(clean_text_func)

# Display the headlines
st.write("### News Headlines")
st.dataframe(df_headlines[['text']])

# Extract the 'Open' price from the fetched data
try:
    open_price_data = fetch_spy_data_func(datetime.datetime.now())
    
    # Check if open_price_data is a list and has at least one element
    if isinstance(open_price_data, list) and len(open_price_data) > 0 and 'Open' in open_price_data[0]:
        st.write("### Opening Price")
        st.write(open_price_data[0]['Open'])
        
        # Prediction button
        if st.button('Predict'):
            # Transform the input
            text_tfidf = vectorizer.transform(df_headlines['cleaned_text'])
            open_price_input = open_price_data[0]['Open']
            
            # Transform the opening price
            open_scaled = scaler.transform(np.array([[open_price_input]]))
            
            # Make predictions
            text_prediction = log_reg.predict(text_tfidf)
            open_prediction = xgb_model.predict(open_scaled)
            
            # Combine predictions
            combined_prediction = (text_prediction[0] + open_prediction[0]) / 2
            final_prediction = int(np.round(combined_prediction))
            
            st.write(f'Predicted Movement: {"Up" if final_prediction == 1 else "Down"}')
    else:
        st.write("Error: No valid data available for 'Open' price.")
except Exception as e:
    st.write(f"An error occurred: {e}")