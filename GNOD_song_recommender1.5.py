#! usr/bin/env python

# import necessary libraries
import requests
import numpy as np 
import pandas as pd
import time
from random import randint
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import spotipy

# get current working directory
cwd = os.getcwd()
print('Current directory is ', cwd)

# connect to spotify with api wrapper
from spotipy.oauth2 import SpotifyClientCredentials

# access spotify web API
string = open(r"C:\Users\Ish\Documents\Ironhack Bootcamp\Unit 6\Day 3\spotAuth.txt","r").read()
cred_dict={}
for line in string.split('\n'):
    if len(line) > 0:
        cred_dict[line.split(':')[0]]=line.split(':')[1]

#Initialize SpotiPy with user credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=cred_dict['CID'],client_secret=cred_dict['CSEC']))

# load pickled models and datasets

Hot_songs = pickle.load(open(r'C:\Users\Ish\Documents\Ironhack Bootcamp\Unit 6\Day 3\GNOD-Song-Recommender\hot_songs_data.pkl','rb'))
song_features = pickle.load(open(r'C:\Users\Ish\Documents\Ironhack Bootcamp\Unit 6\Day 3\GNOD-Song-Recommender\spotify_song_data.pkl','rb'))
Feature_scaler1 = pickle.load(open(r'C:\Users\Ish\Documents\Ironhack Bootcamp\Unit 6\Day 3\GNOD-Song-Recommender\minmaxscaler.pkl','rb'))
kmeans1 = pickle.load(open(r'C:\Users\Ish\Documents\Ironhack Bootcamp\Unit 6\Day 3\GNOD-Song-Recommender\kmeanscluster.pkl','rb'))
birchcluster = pickle.load(open(r'C:\Users\Ish\Documents\Ironhack Bootcamp\Unit 6\Day 3\GNOD-Song-Recommender\birchcluster.pkl','rb'))

# create a function to take in a user request and search for song features based on the request.
def song_seeker(request):
    search = sp.search(q = request['Track Title'], type = 'track' )
    for track in search['tracks']['items']:
        if request['Track Title'].lower()in [track['name'].lower() for track in search['tracks']['items']] and request['Artist Name'].lower() in [track['artists'][x]['name'].lower() for x in range(len(track['artists']))]:
           features = sp.audio_features(track['id'])
           feature_frame = pd.DataFrame(features)
           feature_frame = feature_frame._get_numeric_data()
           feature_frame.drop(['time_signature', 'mode'], axis = 1, inplace = True)
           return feature_frame
        elif request['Track Title'].lower() in [track['name'].lower() for track in search['tracks']['items']] :
           features = sp.audio_features(track['id'])
           feature_frame = pd.DataFrame(features)
           feature_frame = feature_frame._get_numeric_data()
           feature_frame.drop(['time_signature','mode'], axis = 1, inplace = True)
           return feature_frame
        else:
           try_again = f"Sorry, no results found for: {request}.Are you sure you spelled it correctly?"
           return try_again

# create a function to take in a cluster and return a random other song with the same cluster number.
def song_retriever(clusters, song_list = song_features):
    cluster_list1 = song_features[song_features[['Kmeans clusters','Birch clusters']] == [5,4]]
    cluster_list2 = song_features[song_features['Birch clusters'] == 4]
    cluster_list = pd.concat([cluster_list1,cluster_list2])
    cluster_list.dropna(inplace = True)
    random_recommendation = cluster_list.iloc[randint(0,len(cluster_list))]
    random_recommendation = random_recommendation[['song_title', 'artists']]
    return random_recommendation.to_dict()

 
# create a function to search through a dataframe of songs check that the input is in it, and return a random song. 
def random_song(song_data = Hot_songs):
    # Create an input option to allow a user to input a track title and artist name.
    Hot = False
    while not Hot:
        user_song_input = input('Song:')
        user_artist_input = input('Artist: ')
        user_request = {'Track Title':user_song_input, 'Artist Name':user_artist_input}

        for i in song_data['Track Title']:    
            if user_request['Track Title'].lower() == i.lower():
                random_rec = song_data.iloc[randint(0,(song_data.shape[0]))]
                print(f'If you like {user_request} you might like:\n"{random_rec[1]}" by {random_rec[0]}')
                Hot = True
                break
            else:
                pass
        if Hot:
            pass
        else:
            results = song_seeker(user_request)
            song_title = user_request['Track Title']
            artist = user_request['Artist Name']
            if type(results) != str and type(results) != None:
                scaled_results = pd.DataFrame(Feature_scaler1.transform(results))
                k_cluster = kmeans1.predict(scaled_results)[0]
                brch_cluster = birchcluster.predict(scaled_results)[0]
                clusters = [k_cluster,brch_cluster]
                recommendation = song_retriever(clusters,song_features)
                recommendation_title = recommendation['song_title']
                recommendation_artist = recommendation['artists']
                print(f'If you like "{song_title}" then you might like:\n "{recommendation_title}" by {recommendation_artist}')
                Hot = True
                break
            else:
                print(results)

random_song()
        
