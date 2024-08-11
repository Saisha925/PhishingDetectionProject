#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
file_path = r"C:\Users\DELL\Downloads\dataset_phishing (1).csv"
df = pd.read_csv(file_path)

# Display the column names to confirm the text column name
print(df.columns)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize PorterStemmer and WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Perform stemming
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    # Perform lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    return ' '.join(lemmatized_tokens)

# Assuming the correct column name for the text data is 'message'
df['processed_text'] = df['status'].apply(preprocess_text)


# In[33]:


# Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Assuming the target column is named 'label'
y = df['url']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print('Classification Report:')
print(report)

# Perform cross-validation with reduced number of splits
cross_val_scores = cross_val_score(clf, X, y, cv=2, scoring='accuracy')  # Changed cv to 3
print(f'Cross-validation scores: {cross_val_scores}')
print(f'Average cross-validation score: {cross_val_scores.mean()}')


# In[45]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
file_path = r"C:\Users\DELL\Downloads\dataset_phishing (1).csv"
df = pd.read_csv(file_path)


if 'url' not in df.columns:
    print("Please ensure your dataset has a column named 'url'")
else:
    texts = df['url']

    # Define the preprocessing functions
    def preprocess_text(text):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Convert to lower case
        tokens = [word.lower() for word in tokens]
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
        
        # Perform stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        
        # Alternatively, perform lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(lemmatized_tokens)  # Join tokens back into a single string

    # Apply preprocessing to each text entry
    df['cleaned_text'] = texts.apply(preprocess_text)

    # Show a sample of the cleaned text
    print(df[['url', 'cleaned_text']].head(100))

    # Save the cleaned data to a new CSV file
    cleaned_file_path = r"C:\Users\DELL\Downloads\dataset_phishing(clean).csv"
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")


# In[38]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
file_path = r"C:\Users\DELL\Downloads\dataset_phishing (1).csv"
df = pd.read_csv(file_path)


# Define the preprocessing functions
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
        
    # Convert to lower case
    tokens = [word.lower() for word in tokens]
        
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
        
    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
        
    # Alternatively, perform lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
    return ' '.join(lemmatized_tokens)  # Join tokens back into a single string

    # Apply preprocessing to each text entry
    df['cleaned_text'] = texts.apply(preprocess_text)

    # Show a sample of the cleaned text
    print(df[['url', 'cleaned_text']].head())

    # Save the cleaned data to a new CSV file
    cleaned_file_path = r"C:\Users\DELL\Downloads\dataset_phishing.csv"
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")

