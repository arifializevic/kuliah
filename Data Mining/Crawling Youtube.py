import spacy
import matplotlib.pyplot as plt
import pandas as pd
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

nlp = spacy.load("en_core_web_lg")


# Load IndoBERT model and tokenizer
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer)

# Video ID and API Key
video_id = "bER11eIA3y0"  # video ID Youtube
apikey = "AIzaSyDrwVcjd1hQpsivM11bq996l1zn9xj5r38"  # API Key


def video_comments(video_id, apikey):
    """
    Fetch comments from a YouTube video.
    Args:
        video_id (str): YouTube video ID.
        apikey (str): YouTube API key.
    Returns:
        list: A list of comments with metadata.
    """
    replies = []
    youtube = build('youtube', 'v3', developerKey=apikey)
    video_response = youtube.commentThreads().list(
        part='snippet,replies', videoId=video_id).execute()

    while video_response:
        for item in video_response['items']:
            published = item['snippet']['topLevelComment']['snippet']['publishedAt']
            user = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            likeCount = item['snippet']['topLevelComment']['snippet']['likeCount']
            replies.append([published, user, comment, likeCount])

            # Process replies
            replycount = item['snippet']['totalReplyCount']
            if replycount > 0:
                for reply in item.get('replies', {}).get('comments', []):
                    published = reply['snippet']['publishedAt']
                    user = reply['snippet']['authorDisplayName']
                    repl = reply['snippet']['textDisplay']
                    likeCount = reply['snippet']['likeCount']
                    replies.append([published, user, repl, likeCount])

        # Handle pagination
        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                pageToken=video_response['nextPageToken'],
                videoId=video_id
            ).execute()
        else:
            break
    return replies


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


def preprocess_text(text):
    """
    Preprocess text by normalizing and removing stopwords using spaCy.
    Args:
        text (str): Original text.
    Returns:
        str: Cleaned text.
    """
    # Use spaCy to process the text
    doc = nlp(text.lower())  # Convert to lowercase and process with spaCy
    # Filter out stopwords and non-alphabetical tokens
    text = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(text)


# ambil komentar
comments = video_comments(video_id, apikey)

# konvert ke DataFrame
df = pd.DataFrame(comments, columns=[
                  'publishedAt', 'authorDisplayName', 'textDisplay', 'likeCount'])


# Apply preprocessing
df['textDisplay'] = df['textDisplay'].apply(preprocess_text)

# Tambahkan kolom sentiment
df['sentiment'] = df['textDisplay'].apply(analyze_sentiment_indo)
df

# Save to CSV
output_path = 'dataset/youtube_comments.csv'
df.to_csv(output_path, index=False)
print(f"Data successfully saved to '{output_path}'")

# buat grafik plot
sentiment_counts = df['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'count']

plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts['sentiment'], sentiment_counts['count'], color=[
        'green', 'blue', 'red'])
plt.title('Sentiment Distribution: Positive vs Neutral vs Negative', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Number of Comments', fontsize=12)

# Tambahkan grid untuk memperjelas tampilan
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tampilkan grafik
plt.show()
