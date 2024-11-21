import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))

paslon = {
    '1': ['khofifah indar parawansa', 'emil elestianto dardak'],
    '2': ['tri rismaharini', 'zahrul azhar asumta gus hans', 'risma'],
    '3': ['luluk nur hamidah', 'lukmanul khakim']
}

# Load IndoBERT model and tokenizer
model_name = "indobenchmark/indobert-base-p2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer)

# Video ID and API Key
video_id = "gICn_zzf3j4"  # Video ID YouTube
apikey = "AIzaSyDrwVcjd1hQpsivM11bq996l1zn9xj5r38"  # API Key


def video_comments(video_id, apikey):
    replies = []
    youtube = build('youtube', 'v3', developerKey=apikey)
    video_response = youtube.commentThreads().list(
        part='snippet,replies', videoId=video_id).execute()

    while video_response:
        for item in video_response['items']:
            published = item['snippet']['topLevelComment']['snippet']['publishedAt']
            user = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            replies.append([published, user, comment])

            # Process replies
            replycount = item['snippet']['totalReplyCount']
            if replycount > 0:
                for reply in item.get('replies', {}).get('comments', []):
                    published = reply['snippet']['publishedAt']
                    user = reply['snippet']['authorDisplayName']
                    repl = reply['snippet']['textDisplay']
                    replies.append([published, user, repl])

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
    result = sentiment_pipeline(comment)[0]
    label = result['label']
    if label == "LABEL_1":
        return "positif"
    elif label == "LABEL_0":
        return "negatif"
    else:
        return "netral"


def preprocess_text_nltk(text):
    text = text.lower()
    tokens = word_tokenize(text)
    # Remove stopwords and keep only alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()
              and word not in stop_words]
    return ' '.join(tokens)


def find_paslon(text):
    for paslon_number, candidates in paslon.items():
        for candidate in candidates:
            # Cek apakah nama kandidat ada di teks komentar
            if any(name in text for name in candidate.split()):
                return int(paslon_number)
    return 0


comments = video_comments(video_id, apikey)

df = pd.DataFrame(comments, columns=[
                  'publishedAt', 'authorDisplayName', 'textDisplay'])

df['textDisplay'] = df['textDisplay'].apply(preprocess_text_nltk)
df['sentiment'] = df['textDisplay'].apply(analyze_sentiment_indo)
df['paslon'] = df['textDisplay'].apply(find_paslon)

# Save ke CSV
output_path = 'dataset/youtube_comments_pilgub_jatim_2024.csv'
df.to_csv(output_path, index=False)
print(f"Data successfully saved to '{output_path}'")
