import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel,cosine_similarity

spotify_data = pd.read_csv("C:/Users/LENOVO/OneDrive/Desktop/4thsem/projects/ai/SpotifyFeatures.csv")
df = spotify_data
genre_ohe = pd.get_dummies(df.genre)
key_ohe = pd.get_dummies(df.key)

scaled_features = MinMaxScaler().fit_transform(
    [df['acousticness'].values,
     df['danceability'].values,
     df['duration_ms'].values,
     df['energy'].values,
     df['instrumentalness'].values,
     df['liveness'].values,
     df['loudness'].values,
     df['speechiness'].values,
     df['tempo'].values,
     df['valence'].values,
     ]
)


df[['acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence']] = scaled_features.T

df = df.drop('genre',axis = 1)
df = df.drop('artist_name', axis = 1)
df = df.drop('track_name', axis = 1)
df = df.drop('popularity', axis = 1)
df = df.drop('key', axis = 1)
df = df.drop('mode', axis =1)
df = df.drop('time_signature', axis = 1)

df = df.join(genre_ohe)
df= df.join(key_ohe)

scope = 'user-library-read'
token = util.prompt_for_user_token(
    scope,
    client_id = 'ebd02e5566944a8aa1139075058bc52d',
    
    client_secret = '7d3be05ab58743998b2899e3c2815630',
    

    redirect_uri = 'http://localhost:8881/callback'
)

sp = spotipy.Spotify(auth=token)
playlist_dic = {}
playlist_cover_art = {}

for i in sp.current_user_playlists()['items']:
    playlist_dic[i['name']] = i['uri'].split(':')[2]
    playlist_cover_art[i['uri'].split(':')[2]] = i['images'][0]['url']

# print(playlist_dic)


def generate_playlist_df(playlist_name, playlist_dic, df):
    
    playlist = pd.DataFrame()

    for i, j in enumerate(sp.playlist(playlist_dic[playlist_name])['tracks']['items']):
        playlist.loc[i, 'artist'] = j['track']['artists'][0]['name']
        playlist.loc[i, 'track_name'] = j['track']['name']
        playlist.loc[i, 'track_id'] = j['track']['id']
        playlist.loc[i, 'url'] = j['track']['album']['images'][1]['url']
        playlist.loc[i, 'date_added'] = j['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
    
    playlist = playlist[playlist['track_id'].isin(df['track_id'].values)].sort_values('date_added',ascending = False)

    return playlist

playlist_df = generate_playlist_df('Vibe', playlist_dic, df)

def visualize_cover_art(playlist_df):
    temp = playlist_df['url'].values
    plt.figure(figsize=(15,int(1.2 * len(temp))), facecolor='#be03fc')
    columns = 6
    
    for i, url in enumerate(temp):
        plt.subplot(int(len(temp) / columns + 1 ), columns, i + 1)
        image = io.imread(url)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        s='' 
        plt.xlabel(s.join(playlist_df['track_name'].values[i].split(' ')[:4]), fontsize = 8, fontweight='bold')
        plt.tight_layout(h_pad=0.9, w_pad=0)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.show()

    

# visualize_cover_art(playlist_df)

def generate_playlist_vector(df, playlist_df, weight_factor):
    df1 = df[df['track_id'].isin(playlist_df['track_id'].values)]
    df1 = df1.merge(playlist_df[['track_id', 'date_added']], on = 'track_id', how = 'inner')
    df2 = df[-df['track_id'].isin(playlist_df['track_id'].values)]
    playlist_feature_set = df1.sort_values('date_added',ascending=False)
    
    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'days_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days)
        
    
    playlist_feature_set['weight'] = playlist_feature_set['days_from_recent'].apply(lambda x: weight_factor ** (-x))
    
    playlist_feature_set_weighted = playlist_feature_set.copy()
    
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-3].mul(playlist_feature_set_weighted.weight.astype(int),0))   
    
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-3]

    return playlist_feature_set_weighted_final.sum(axis = 0), df2

playlist_vector, nonplaylist_df = generate_playlist_vector(df, playlist_df, 1.2)
print(playlist_vector.shape)
print(nonplaylist_df.head())


def generate_recommendation(df, playlist_vector, nonplaylist_df):

    non_playlist = df[df['track_id'].isin(nonplaylist_df['track_id'].values)]
    non_playlist['sim'] = cosine_similarity(nonplaylist_df.drop(['track_id'], axis = 1).values, playlist_vector.drop(labels = 'track_id').values.reshape(1, -1))[:,0]
    non_playlist_top25 = non_playlist.sort_values('sim',ascending = False).head(25)
    non_playlist_top25['url'] = non_playlist_top25['track_id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    
    return  non_playlist_top25

top25 = generate_recommendation(spotify_data, playlist_vector, nonplaylist_df)  
print(top25)

visualize_cover_art(top25)