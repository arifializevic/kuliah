import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))

# Load IndoBERT model and tokenizer
model_name = "indolem/indobert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer)

# Video ID and API Key
video_id = "bER11eIA3y0"  # Video ID YouTube
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
    # Remove stopwords and non-alphabetical tokens
    words = [word for word in words if word.isalpha()
             and word not in stop_words]
    return ' '.join(words)


comments = video_comments(video_id, apikey)

df = pd.DataFrame(comments, columns=[
                  'publishedAt', 'authorDisplayName', 'textDisplay', 'likeCount'])

df['textDisplay'] = df['textDisplay'].apply(preprocess_text_nltk)
df['sentiment'] = df['textDisplay'].apply(analyze_sentiment_indo)
print(df)

# Save ke CSV
# output_path = 'dataset/youtube_comments.csv'
# df.to_csv(output_path, index=False)
# print(f"Data successfully saved to '{output_path}'")

plt.figure(figsize=(8, 10))
sns.countplot(data=df['authorDisplayName'],
              order=df['authorDisplayName'].value_counts().index)

# grafik positive vs neutral vs negative
sentiment_counts = df['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'count']
unique_sentiments = sentiment_counts['sentiment'].nunique()
color_palette = sns.color_palette("RdYlGn", unique_sentiments)

# Plot menggunakan Seaborn
sns.barplot(
    data=sentiment_counts,
    x='sentiment',
    y='count',
    hue='sentiment',  # Menambahkan hue untuk kategori
    palette=color_palette
)

# Menambahkan judul dan label sumbu
sns.despine()  # Menghilangkan garis tepi yang tidak perlu
plt.title('Sentiment Distribution: Positive vs Neutral vs Negative', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Number of Comments', fontsize=12)

# Tampilkan grafik
plt.show()
