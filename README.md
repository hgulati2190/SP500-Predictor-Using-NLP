# SP500-Predictor-Using-NLP
## **Objective:** 
Leverage Natural Language Processing (**NLP**) to analyze news **headlines** to provide valuable insights for predicting **stock market trends**.
## **Scope of Work:**

**Data Collection & Preparation:** Prepare training data by concatenating various headlines from CNBC, Guardian and Reuters on the same day. Add SPY information seven days after the gathered headline date.

**Feature Analysis & Model Training:** Use prepped data to finetune a Binary Classification AI Model pre-trained on financial text.

**Model Selection & Explanation:** Setup SHAP to explain features of the fine tuned model. Train a spaCy model on Named Entity Recognition (NER)

**Web App Cloud Deployment:** Build a Web App for user interactivity and deploy to the cloud

## **Tools & Technologies:**
**Transformers:** package in Python, developed by Hugging Face, provides state-of-the-art natural language processing models and tools for tasks such as text classification, translation, question answering, and more, facilitating easy integration and use of transformer-based models.

**yfinance:** package in Python allows users to easily download and query historical market data from Yahoo Finance, providing a convenient interface for financial data analysis and research. 

**Shapley Values:** Based on game theory and belongs to Transferable Utility which consists of a Player Set (Word Vocabulary) and a Characteristic Function (Stock Movement Prediction) to assign an importance value to each player.

**SHAP:** SHapley Additive exPlanations is a Python package that provides a unified approach to explain the output of machine learning models by assigning each feature an importance value for a particular prediction based on cooperative game theory principles.

**spaCy:** A powerful and fast Python library for advanced natural language processing tasks, including tokenization, part-of-speech tagging, named entity recognition, and dependency parsing.

**NLP:** Natural Language Processing focuses on the interaction between computers and humans through natural language, enabling computers to understand, interpret, and generate human language.

**NER:** Named Entity Recognition is a natural language processing task that involves identifying and classifying proper nouns or entities in text into predefined categories such as names of Companies, Political Groups, People, and Stock Exchanges.

**AWS:** For web application
