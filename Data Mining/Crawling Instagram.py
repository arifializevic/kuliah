import instaloader
import nltk
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))

# Load IndoBERT model and tokenizer
model_name = "indolem/indobert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer)

L = instaloader.Instaloader()


profile_name = "tempo"
profile = instaloader.Profile.from_username(L.context, profile_name)


def preprocess_text_nltk(text):
    """
    Preprocess text by normalizing and removing stopwords using NLTK.
    Args:
        text (str): Original text.
    Returns:
        str: Cleaned text.
    """
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


def analyze_sentiment_indo(comment):
    """
    Analyze sentiment using IndoBERT.
    Args:
        comment (str): Text to analyze.
    Returns:
        str: Sentiment label ('positif', 'negatif', or 'netral').
    """
    result = sentiment_pipeline(comment)[0]
    label = result['label']
    if label == "LABEL_1":
        return "positif"
    elif label == "LABEL_0":
        return "negatif"
    else:
        return "netral"


# Extract posts and comments from Instagram profile
comments_data = []

for post in profile.get_posts():
    # Iterate through each post's comments (if any)
    for comment in post.get_comments():
        user = comment.owner.username
        text = comment.text
        comment_date = comment.created_at_utc
        likes = comment.likes
        sentiment = analyze_sentiment_indo(preprocess_text_nltk(text))

        comments_data.append([comment_date, user, text, likes, sentiment])

# Create DataFrame for the collected data
df_instagram = pd.DataFrame(comments_data, columns=[
                            'publishedAt', 'authorDisplayName', 'textDisplay', 'likeCount', 'sentiment'])

# Save the data to CSV
output_path = 'dataset/instagram_comments.csv'
df_instagram.to_csv(output_path, index=False)
print(f"Data successfully saved to '{output_path}'")
