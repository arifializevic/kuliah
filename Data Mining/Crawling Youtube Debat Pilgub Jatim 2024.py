import gdown
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))

# Load IndoBERT model and tokenizer
model_name = "indobenchmark/indobert-large-p2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer)

# Ambil file GDrive
file_path = "paslon.xlsx"
file_id = '19Xe7xVModP0V59gsjy6IA1FUakFcGjmq'
gdown.download(
    f'https://drive.google.com/uc?export=download&id={file_id}', file_path, quiet=False)
paslon_df = pd.read_excel(file_path)

# Group kandidat berdasarkan paslon
grouped_paslon = paslon_df.groupby(
    'paslon')['kandidat'].apply(list).reset_index()
paslon = grouped_paslon.set_index('paslon')['kandidat'].to_dict()

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
output_path = 'youtube_comments_pilgub_jatim_2024.csv'
df.to_csv(output_path, index=False)
print(f"Data successfully saved to '{output_path}'")


### Membuat Word Cloud ###
text = ' '.join(df['textDisplay'])
wordcloud = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='viridis',
    max_words=100
).generate(text)

# Plot Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hilangkan sumbu
plt.title('Word Cloud Paslon', fontsize=16)
plt.show()


### Membuat BarPlot ###
# Hitung jumlah komentar berdasarkan sentimen
sentiment_counts = df['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'count']

# Visualisasi menggunakan Seaborn
sns.barplot(
    data=sentiment_counts,
    x='sentiment',
    y='count',
    palette='viridis'
)

# Tambahkan detail visualisasi
sns.despine()
plt.title('Distribusi Sentimen Komentar', fontsize=16)
plt.xlabel('Sentimen', fontsize=12)
plt.ylabel('Jumlah Komentar', fontsize=12)
plt.show()


# Filter komentar dengan sentimen positif
positive_comments = df[df['sentiment'] == 'positif']

# Hitung jumlah komentar positif untuk setiap paslon
positive_counts = positive_comments['paslon'].value_counts().reset_index()
positive_counts.columns = ['paslon', 'count']

# Sort berdasarkan paslon untuk memastikan urutan
positive_counts = positive_counts.sort_values(by='paslon')

sns.barplot(
    data=positive_counts,
    x='paslon',
    y='count',
    hue='paslon',
    dodge=False
)

sns.despine()
plt.title('Sentimen Positif dari Paslon', fontsize=16)
plt.xlabel('Paslon', fontsize=12)
plt.ylabel('Total Komentar Positif', fontsize=12)
plt.legend([], [], frameon=False)
plt.show()


# Filter komentar dengan sentimen negatif
negative_comments = df[df['sentiment'] == 'negatif']

# Hitung jumlah komentar negatif untuk setiap paslon
negative_counts = negative_comments['paslon'].value_counts().reset_index()
negative_counts.columns = ['paslon', 'count']

# Sort berdasarkan paslon untuk memastikan urutan
negative_counts = negative_counts.sort_values(by='paslon')

sns.barplot(
    data=negative_counts,
    x='paslon',
    y='count',
    hue='paslon',
    dodge=False
)

sns.despine()
plt.title('Sentimen Negatif dari Paslon', fontsize=16)
plt.xlabel('Paslon', fontsize=12)
plt.ylabel('Total Komentar Negatif', fontsize=12)
plt.legend([], [], frameon=False)
plt.show()


# Filter komentar dengan sentimen netral
neutral_comments = df[df['sentiment'] == 'netral']

# Hitung jumlah komentar positif untuk setiap paslon
neutral_counts = neutral_comments['paslon'].value_counts().reset_index()
neutral_counts.columns = ['paslon', 'count']

# Sort berdasarkan paslon untuk memastikan urutan
neutral_counts = neutral_counts.sort_values(by='paslon')

sns.barplot(
    data=neutral_counts,
    x='paslon',
    y='count',
    hue='paslon',
    dodge=False
)

sns.despine()
plt.title('Sentimen Netral dari Paslon', fontsize=16)
plt.xlabel('Paslon', fontsize=12)
plt.ylabel('Total Komentar Netral', fontsize=12)
plt.legend([], [], frameon=False)
plt.show()
