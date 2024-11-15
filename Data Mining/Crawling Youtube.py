import pandas as pd
from googleapiclient.discovery import build


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
            likeCount = item['snippet']['topLevelComment']['snippet']['likeCount']

            replies.append([published, user, comment, likeCount])

            replycount = item['snippet']['totalReplyCount']

            if replycount > 0:
                for reply in item['replies']['comments']:

                    # Extract reply
                    published = reply['snippet']['publishedAt']
                    user = reply['snippet']['authorDisplayName']
                    repl = reply['snippet']['textDisplay']
                    likeCount = reply['snippet']['likeCount']

                    # replies.append(reply)
                    replies.append([published, user, repl, likeCount])

            # print comment with list of reply
            # print(comment, replies, end = '\n\n')

            # empty reply list
            # replies = []

        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                pageToken=video_response['nextPageToken'],
                videoId=video_id
            ).execute()
        else:
            break
    return replies


# Video id nya
# https://www.youtube.com/watch?v=lWp4r4Y2A58
video_id = "lWp4r4Y2A58"
apikey = 'AIzaSyDrwVcjd1hQpsivM11bq996l1zn9xj5r38'

# jalankan function
comments = video_comments(video_id, apikey)
df = pd.DataFrame(comments, columns=[
                  'publishedAt', 'authorDisplayName', 'textDisplay', 'likeCount'])

# simpan ke CSV
df.to_csv('dataset\youtube_comments.csv', index=False)
