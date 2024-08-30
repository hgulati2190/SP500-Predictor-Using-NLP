pip install streamlit shap xgboost transformers
import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Load the model
model_path = "./xgb_best_model_hemant.pkl"  
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(model_path)

# Load the tokenizer and sentiment model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Load SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

def analyze_sentiment(headline):
    inputs = tokenizer(headline, return_tensors="tf", truncation=True, padding=True)
    outputs = sentiment_model(inputs.input_ids)
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    sentiment_score = probabilities[0][1].numpy()  # Assuming positive sentiment is the second class
    return sentiment_score

# Streamlit app
st.title("Stock Price Prediction Based on News Headlines")

# User inputs for headlines
user_input = st.text_area("Enter news headlines:")

if st.button("Predict"):
    # Perform sentiment analysis
    sentiment_score = analyze_sentiment(user_input)
    
    # Assume some example data for other features (e.g., Close price)
    example_close_price = 400  # This should be dynamic based on the stock data
    example_variation = 0.5    # Same here, dynamically based on stock data
    
    # Prepare the input features
    input_data = pd.DataFrame({
        "Sentiment_Score": [sentiment_score],
        "Close": [example_close_price],
        "Variation": [example_variation]
    })
    
    # Make a prediction
    prediction = xgb_model.predict(input_data)
    
    # Display the prediction
    label = "Increase" if prediction[0] == 1 else "Decrease"
    st.write(f"Predicted Stock Movement: {label}")

    # SHAP explanation
    shap_values = explainer.shap_values(input_data)
    st.subheader("SHAP Explanation")
    shap.initjs()
    st_shap = shap.force_plot(explainer.expected_value, shap_values, input_data, show=False)
    st.pyplot(st_shap)