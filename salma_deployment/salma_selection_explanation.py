import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import shap
import nltk
import io
import streamlit.components.v1 as components
import os


base_dir = os.path.dirname(__file__)
log_reg_path = os.path.join(base_dir, 'salma_dll', 'salma_log_reg.pkl')
with open(log_reg_path, 'rb') as file:
    log_reg = pickle.load(file)

# Load the models and transformers
with open('salma_deployment/salma_dll/salma_log_reg.pkl', 'rb') as file:
    log_reg = pickle.load(file)

with open('salma_deployment/salma_dll/salma_xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

with open('salma_deployment/salma_dll/salma_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('salma_deployment/salma_dll/salma_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    


# In[6]:


# Load functions
def load_functions(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
nltk.download('punkt_tab')
nltk.download('punkt', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')

# Add the path to nltk data
nltk.data.path.append('nltk_data')

functions_dict = load_functions('salma_deployment/salma_dll/salma_functions.pkl')
print("Functions Loaded:", functions_dict)

clean_text_func = functions_dict['clean_text']
collect_headlines_func = functions_dict['collect_headlines']
fetch_spy_data_func = functions_dict['fetch_spy_data']


# In[8]:


# Streamlit app layout
st.title('Stock Movement Prediction')

# User inputs
# Convert to DataFrame
df_headlines = pd.DataFrame(collect_headlines_func())
# Apply clean_text function to the 'text' column
df_headlines['cleaned_text'] = df_headlines['text'].apply(clean_text_func)


# In[126]:


# Select the 'text' column
text_input = df_headlines['cleaned_text']


open_price_data = fetch_spy_data_func(datetime.datetime.now())
text_tfidf = vectorizer.transform(text_input)


# Display the headlines
st.write("### News Headlines")
st.dataframe(df_headlines[['text']])
# In[128]:


# Extract the 'Open' price from the fetched data
try:
    # Check if open_price_data is a list and has at least one element
    if isinstance(open_price_data, list) and len(open_price_data) > 0 and 'Open' in open_price_data[0]:
        date_str = open_price_data[0]['Date']
        if isinstance(date_str, pd.Timestamp):
            date_str = date_str.strftime('%Y-%m-%d')  # Format the date as needed
            open_price_str = str(open_price_data[0]['Open'])
        st.write(date_str+ ' - Opening Price   ' + open_price_str)
        feature_names = vectorizer.get_feature_names_out()
        # Prediction button
        if st.button('Predict'):
            # Transform the input
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






# In[144]:


# Feature Importance for Logistic Regression
coefficients = log_reg.coef_.flatten()
importance = np.abs(coefficients)
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Plot feature importance for Logistic Regression
fig, ax = plt.subplots(figsize=(10, 8))
importance_df.head(20).plot(kind='barh', x='Feature', y='Importance', ax=ax, legend=False)
ax.set_title('Top 20 Feature Importances for Logistic Regression - From Original Model')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')

# Display the plot in Streamlit
st.pyplot(fig)

# In[146]:

st.write("### Word Frequency - Nothing to do with our Model")
result=""
result= result.join(text_input)


# In[148]:


# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(result)

# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')

# Display the word cloud in Streamlit
st.pyplot(fig)


# In[158]:


# Create the explainer with the summarized background data
explainer_log = shap.LinearExplainer(log_reg, text_tfidf)


# In[160]:


# Compute SHAP values for the first 100 samples of the test data
shap_values_log = explainer_log.shap_values(text_tfidf)


# In[164]:


# Create a buffer to save the plot
buf = io.BytesIO()

# Plot the summary of SHAP values
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values_log, text_tfidf, feature_names=vectorizer.get_feature_names_out(), show=False)
plt.savefig(buf, format='png')
buf.seek(0)
# In[174]:
# Display the SHAP summary plot in Streamlit
st.image(buf, use_column_width=True, caption='SHAP Summary Plot')


# Streamlit app layout
st.title('SHAP Force Plot')

# Input for instance index
instance_index = st.number_input(
    'Enter the instance index (0 to {})'.format(len(df_headlines) - 1),
    min_value=0,
    max_value=max(len(df_headlines) - 1, 10),
    value=10,
    step=1
)

# Ensure the instance_index does not exceed the number of samples
instance_index = min(instance_index, len(df_headlines) - 1)

# Compute SHAP values for the selected instance
shap_values_instance = explainer_log.shap_values(text_tfidf[instance_index].toarray())

# Generate and save the SHAP force plot as a static image
try:
    shap.force_plot(
        explainer_log.expected_value,
        shap_values_instance,
        text_tfidf[instance_index].toarray(),
        feature_names=vectorizer.get_feature_names_out(),
        matplotlib=True  # Optionally use matplotlib rendering
    )
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Display the plot in Streamlit
    st.image(buf, caption='SHAP Force Plot', use_column_width=True)
except Exception as e:
    st.write(f"An error occurred: {e}")


