import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import re
import joblib
import string
import matplotlib.pyplot as plt
import nltk
import matplotlib.colors as mcolors
import os
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

st.set_page_config(layout='centered', page_title='Sentiment Analysis')

# Navigation buttons at the top
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Home"):
        st.switch_page("Home.py")

with col2:
    if st.button("Kmeans Clustering"):
        st.switch_page("pages/KMEANS.py")

with col3:
    if st.button("Association Rules"):
        st.switch_page("pages/Association_Rules.py")

with col4:
    if st.button("Sentiment Analysis"):
        st.switch_page("pages/Sentiment_Analysis.py")

st.markdown("<h1 style='text-align: center;'>Sentiment Analysis</h1>", unsafe_allow_html=True)

# Load datasets
df_implementation = pd.read_csv('data/test.csv', encoding='Windows-1252')
model_file = 'model/Naive_Bayes_model.joblib'
vectorizer_file = 'model/vectorizer.joblib'

# Function to load model and vectorizer
@st.cache_data
def load_model_vectorizer():
    # Load and return both the Naive Bayes model and the vectorizer
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    return model, vectorizer

# Function to handle sentiment analysis for the uploaded dataset
def sentiment_analysis(df_implementation): 
    model, vectorizer = load_model_vectorizer()
    # Predictions for the implementation dataset
    X_implementation = vectorizer.transform(df_implementation['comment'])
    df_implementation['predicted_sentiment'] = model.predict(X_implementation)

    # Get confidence scores
    proba_implementation = model.predict_proba(X_implementation)
    confidence_scores = proba_implementation.max(axis=1)  # Confidence of the predicted class
    df_implementation['accuracy'] = confidence_scores * 100  # Convert to percentage
    
    # Calculate classification report and accuracy score
    y_true = df_implementation['sentiment']
    y_pred = df_implementation['predicted_sentiment']
    
    accuracy = accuracy_score(y_true, y_pred)

    return df_implementation, vectorizer, model, accuracy, y_true, y_pred

# data preprocessing
def pre_process(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # Remove repeated words
    cleaned_comment = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return cleaned_comment

# Updated pre_process function with lemmatization
def preprocess_text_column(dataframe, column_name):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars and numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        tokens = word_tokenize(text)  # Tokenize text
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
        return ' '.join(tokens)
    
    dataframe[column_name] = dataframe[column_name].apply(clean_text)
    return dataframe

def count_words_by_sentiment(df, vectorizer):
    # Transform comments to token counts
    X = vectorizer.transform(df['comment'])
    
    # Create a DataFrame from the transformed data
    word_counts_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Add sentiment column
    word_counts_df['sentiment'] = df['sentiment']
    
    # Initialize counters
    positive_counts = word_counts_df[word_counts_df['sentiment'] == 'positive'].drop(columns='sentiment').sum()
    negative_counts = word_counts_df[word_counts_df['sentiment'] == 'negative'].drop(columns='sentiment').sum()
    
    return positive_counts, negative_counts

# Function to display output
def display_output(df):
    df_filtered = df.drop(columns=['sentiment', 'language']) # to remove from displaying the sentiment and language column
    st.write(df_filtered)

# Function to predict sentiment for user input
def predict_sentiment(user_comment, vectorizer, model):
    user_comment_transformed = vectorizer.transform([user_comment])
    predicted_sentiment = model.predict(user_comment_transformed)[0]
    sentiment_accuracy = model.predict_proba(user_comment_transformed).max(axis=1)[0] * 100

    return predicted_sentiment, sentiment_accuracy

# Function to display classification report
def display_classification_report(y_true, y_pred):
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, labels=['positive', 'negative'], output_dict=True)
    st.dataframe(report)

    # Display Accuracy score
    st.write(f"**Accuracy Score:** {accuracy*100:.2f}%")

# Display Confusion matrix
def display_confusion_matrix(y_true, y_pred):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['positive', 'negative'])
    
    # Normalize the confusion matrix by row (true classes) to get percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    st.subheader("Confusion Matrix")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot the confusion matrix with counts
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Positive', 'Negative'])
    disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)  # Using 'Blues' colormap for better contrast
    
    # Get colormap and normalization
    cmap = plt.get_cmap('Blues')
    norm = mcolors.Normalize(vmin=cm.min(), vmax=cm.max())
    
    # Overlay percentage values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Get the background color intensity at the current cell using normalization
            background_color_intensity = norm(cm[i, j])
            background_color = cmap(background_color_intensity)  # RGBA color
            
            # Calculate brightness to determine text color
            r, g, b, _ = background_color
            brightness = (r * 0.299 + g * 0.587 + b * 0.114)  # Perceived brightness
            
            # Set text color: White for darker background, black for lighter background
            text_color = "white" if brightness < 0.5 else "black"
            
            # Add percentage text
            percentage_text = f'{cm_percentage[i, j]:.2f}%'
            ax.text(j, i + 0.2, percentage_text, ha='center', va='center', color=text_color, fontsize=8)
    
    # Customize plot appearance
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Render the plot in Streamlit
    st.pyplot(fig)

df = preprocess_text_column(df_implementation, 'comment')

# Global model and vectorizer variables
result_df, vectorizer, model, accuracy, y_true, y_pred = sentiment_analysis(df)

# Count unigram words for positive and negative sentiments
positive_word_counts, negative_word_counts = count_words_by_sentiment(result_df, vectorizer)

# Convert word counts to DataFrames for better visualization
word_count_positive_df = pd.DataFrame(positive_word_counts.items(), columns=['Word', 'Count'])
word_count_negative_df = pd.DataFrame(negative_word_counts.items(), columns=['Word', 'Count'])

# Sort and select the top 20 words for each sentiment class
top_positive_words = word_count_positive_df.sort_values(by='Count', ascending=False).head(20)
top_negative_words = word_count_negative_df.sort_values(by='Count', ascending=False).head(20)

# Display Dataset, Data reports, and user input sentiment prediction
tab1, tab2, tab3 = st.tabs(['Dataset','Data Report', 'Try Sentiment Prediction'])

with tab1:
    # Display the analysis result
    display_output(result_df)
    
with tab2, st.container(border=True):
    col1, col2, col3 = st.columns(3)
    container = st.container(border=True)

    # Display classification report, accuracy score, and confusion matrix
    display_confusion_matrix(y_true, y_pred)
    colu1, colu2 = st.columns(2)
    with colu1:
        st.subheader("Top 20 Words - Positive Sentiment:")
        st.write()
        # Apply background gradient styling and display the DataFrame
        styled_positive_words = top_positive_words.style.background_gradient(cmap='Greens')
        st.dataframe(styled_positive_words, use_container_width=True)
    with colu2:
        # Display the top words for each sentiment class - Negative
        st.subheader("\nTop 20 Words - Negative Sentiment:")
        st.write()
        # Apply background gradient styling and display the DataFrame
        styled_negative_words = top_negative_words.style.background_gradient(cmap='Reds')
        st.dataframe(styled_negative_words, use_container_width=True)

    display_classification_report(y_true, y_pred)
    
with tab3:
    # User input for live sentiment prediction
    MIN_WORD_COUNT = 5
    st.markdown("### Predict Sentiment for Your Own Comment")
    st.info(f'Enter a comment with atleast {MIN_WORD_COUNT} words.', icon="\u2139\ufe0f")
    user_comment = st.text_input("Enter a comment:").lower()
    if user_comment:
        cleaned_comment = pre_process(user_comment)
        word_count = len(cleaned_comment.split())  # Count words in cleaned comment

        if word_count < MIN_WORD_COUNT:
            st.warning(f"Please enter at least {MIN_WORD_COUNT} words.")
        else:
            predicted_sentiment, accuracy = predict_sentiment(cleaned_comment, vectorizer, model)
            st.write(f"**Predicted Sentiment:** {predicted_sentiment}")
            st.write(f"**Accuracy:** {accuracy:.2f}%")
            
            # Create a new row for the DataFrame
            new_row = pd.DataFrame({
                'comment': [user_comment],
                'sentiment': [predicted_sentiment],
                'language' : "English/Filipino"
            })
            
            # Append the new row to the existing DataFrame
            df_implementation = pd.concat([df_implementation, new_row], ignore_index=True)

            # Save the updated DataFrame back to the CSV file
            #df_implementation.to_csv('data/test.csv', encoding='Windows-1252', index=False)
            #st.success("Comment and sentiment added to CSV!")
