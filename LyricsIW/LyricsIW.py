##
# This is the main set of code - dataset gathering and processing.
## 

from requests.models import CaseInsensitiveDict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import json
import ast
import numpy as np

import nltk 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer #pip install sklearn does not work for this, use pip install scikit-learn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import string
from lyricsgenius import Genius
from lyricsgenius import API
os.environ["SPOTIPY_CLIENT_ID"] = "YOUR CLIENT ID GOES HERE"
os.environ["SPOTIPY_CLIENT_SECRET"] = "YOUR CLIENT SECRET GOES HERE"
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

errs = 0

#result of call to spotify.recommendation_genre_seeds()
ORIGINAL_GENRE_SEEDS = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 
                        'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 
                        'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 
                        'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 
                        'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore',
                       'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop',
                      'holidays', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 
                      'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 
                      'metal', 'metal-misc', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release', 'opera', 
                      'pagode', 'party', 'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 
                      'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae', 'reggaeton',
                     'road-trip', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 
                     'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish',
                    'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'work-out',
                   'world-music']

#removed all those unlikely to have english lyrics
GENRE_SEEDS = ['acoustic', 'alt-rock', 'alternative', 'black-metal', 'bluegrass', 'blues', 
               'breakbeat', 'british', 'chicago-house', 'children', 'chill', 'club',
              'country', 'dance', 'dancehall', 'death-metal', 'disco',
             'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk',
            'funk', 'garage', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy',
           'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 
           'house', 'idm', 'indie', 'indie-pop', 'industrial', 'jazz', 'kids', 'metal', 'metal-misc',
          'metalcore', 'movies', 'new-release',
         'party',  'pop', 'pop-film', 'post-dubstep', 'power-pop', 
         'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae',
         'road-trip', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad',
         'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'study', 
         'summer', 'synth-pop', 'techno', 'trance', 'trip-hop', 'work-out']




os.environ["GENIUS_ACCESS_TOKEN"] = "YOUR GENIUS ACCESS TOKEN GOES HERE"
genius = Genius()
g_api = API(os.environ["GENIUS_ACCESS_TOKEN"])
genius = Genius(remove_section_headers=True,retries=8)
genius.verbose = True

import sqlite3
dbname = 'Lyrics_DB.sqlite'
conn = sqlite3.connect(dbname)
cur = conn.cursor()

#populates tables in the formatting used by the rest of the program
def create_tables_only():
    try:
        cur.execute('CREATE TABLE SongData (SpotifyID INT PRIMARY KEY, SongName VARCHAR, MainArtistName VARCHAR, ReleaseYear INT, MainGenre VARCHAR, PopularityOnQuery DOUBLE, Acousticness DOUBLE, Danceability DOUBLE, Duration INT, Energy DOUBLE, MajorKey BOOLEAN, Speechiness DOUBLE, Tempo INT, Valence DOUBLE)')
        conn.commit()
    except:
        pass
    try:
        cur.execute('CREATE TABLE FullLyricData (SpotifyID INT PRIMARY KEY, GeniusID INT, FullLyrics VARCHAR)')
        conn.commit()
    except:
        pass
    try:
        cur.execute('CREATE TABLE ProcessedSongData (SpotifyID INT PRIMARY KEY, Sentiment VARCHAR, OriginalCharCount INT, TokenCount INT, UniqueTokenCount INT, FreqDist VARCHAR)')
        conn.commit()
    except:
        pass
    try:
        cur.execute('CREATE TABLE GenreMetadata (Genre VARCHAR PRIMARY KEY, OverallTokenCount INT, GenreUniqueTokenCount INT, NumSongs INT, MeanTokenCount DOUBLE, MeanUniqueTokenCount DOUBLE, OverallFreqDist VARCHAR, MeanPopularityOnQuery DOUBLE, MeanAcousticness DOUBLE, MeanDanceability DOUBLE, MeanDuration DOUBLE, MeanEnergy DOUBLE, RatioInMajorKey DOUBLE, MeanSpeechiness DOUBLE, MeanTempo DOUBLE, MeanValence DOUBLE)')
        conn.commit()
    except:
        pass
    try:
        cur.execute('CREATE TABLE OverallMetadata (Method VARCHAR, OverallTokenCount INT, OverallUniqueTokenCount INT, NumSongs INT, MeanTokenCount DOUBLE, MeanUniqueTokenCount DOUBLE, OverallFreqDist VARCHAR, MeanPopularityOnQuery DOUBLE, MeanAcousticness DOUBLE, MeanDanceability DOUBLE, MeanDuration DOUBLE, MeanEnergy DOUBLE, RatioInMajorKey DOUBLE, MeanSpeechiness DOUBLE, MeanTempo DOUBLE, MeanValence DOUBLE)')
        conn.commit()
    except:
        pass

#Clears out existing tables and recreates them in the specified formatting
def clear_tables():
    
    cur.execute('DROP TABLE IF EXISTS SongData')
    conn.commit()
    cur.execute('CREATE TABLE SongData (SpotifyID INT, SongName VARCHAR, MainArtistName VARCHAR, ReleaseYear INT, MainGenre VARCHAR, PopularityOnQuery DOUBLE, Acousticness DOUBLE, Danceability DOUBLE, Duration INT, Energy DOUBLE, MajorKey BOOLEAN, Speechiness DOUBLE, Tempo INT, Valence DOUBLE)')
    conn.commit()
    cur.execute('DROP TABLE IF EXISTS FullLyricData')
    conn.commit()
    cur.execute('CREATE TABLE FullLyricData (SpotifyID INT, GeniusID INT, FullLyrics VARCHAR)')
    conn.commit()
    cur.execute('DROP TABLE IF EXISTS ProcessedSongData')
    conn.commit()
    cur.execute('CREATE TABLE ProcessedSongData (SpotifyID INT, Sentiment VARCHAR, OriginalCharCount INT, TokenCount INT, UniqueTokenCount INT, FreqDist VARCHAR)')
    conn.commit()
    cur.execute('DROP TABLE IF EXISTS GenreMetadata')
    conn.commit()
    cur.execute('CREATE TABLE GenreMetadata (Genre VARCHAR, OverallTokenCount INT, GenreUniqueTokenCount INT, NumSongs INT, MeanTokenCount DOUBLE, MeanUniqueTokenCount DOUBLE, OverallFreqDist VARCHAR, MeanPopularityOnQuery DOUBLE, MeanAcousticness DOUBLE, MeanDanceability DOUBLE, MeanDuration DOUBLE, MeanEnergy DOUBLE, RatioInMajorKey DOUBLE, MeanSpeechiness DOUBLE, MeanTempo DOUBLE, MeanValence DOUBLE)')
    conn.commit()
    cur.execute('DROP TABLE IF EXISTS OverallMetadata')
    conn.commit()
    cur.execute('CREATE TABLE OverallMetadata (Method VARCHAR, OverallTokenCount INT, OverallUniqueTokenCount INT, NumSongs INT, MeanTokenCount DOUBLE, MeanUniqueTokenCount DOUBLE, OverallFreqDist VARCHAR, MeanPopularityOnQuery DOUBLE, MeanAcousticness DOUBLE, MeanDanceability DOUBLE, MeanDuration DOUBLE, MeanEnergy DOUBLE, RatioInMajorKey DOUBLE, MeanSpeechiness DOUBLE, MeanTempo DOUBLE, MeanValence DOUBLE)')
    conn.commit()

#Tests whether there are any elements in table where the value of col_name is key
#For example, if there are any elements in SongData where the value of SpotifyID is 123456
def key_in_table(key, col_name, table):
    cur.execute('SELECT 1 FROM '+table+' WHERE '+col_name+'="'+key+'"')
    data = cur.fetchall()
    return len(data)==1


#The following set of functions use queries to add to the 5 tables used by the program
#See below for column types
#Generally, SongData and FullLyricData information are collected when the song is queried
#The code goes back later to produce proper rows in ProcessedSongData
#Then back to produce genre-level averages in GenreMetadata, these are useful for metadata but not displayed in the program
#Then looks through GenreMetadata to produce whole-database statistics in OverallMetadata

def add_to_songdata(SpotifyID, SongName, MainArtistName, ReleaseYear, MainGenre, PopularityOnQuery, Acousticness, Danceability, Duration, Energy, MajorKey, Speechiness, Tempo, Valence):
    cur.execute('INSERT INTO SongData (SpotifyID, SongName, MainArtistName, ReleaseYear, MainGenre, PopularityOnQuery, Acousticness, Danceability, Duration, Energy, MajorKey, Speechiness, Tempo, Valence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
                (SpotifyID, SongName, MainArtistName, ReleaseYear, MainGenre, PopularityOnQuery, Acousticness, Danceability, Duration, Energy, MajorKey, Speechiness, Tempo, Valence))
    conn.commit()

def add_to_fulllyricdata(SpotifyID, GeniusID, FullLyrics):
    cur.execute('INSERT INTO FullLyricData (SpotifyID, GeniusID, FullLyrics) VALUES (?, ?, ?)', 
                (SpotifyID, GeniusID, FullLyrics))
    conn.commit()

def add_to_processedsongdata(SpotifyID, Sentiment, OriginalCharCount, TokenCount, UniqueTokenCount, FreqDist):
    cur.execute('INSERT INTO ProcessedSongData (SpotifyID, Sentiment, OriginalCharCount, TokenCount, UniqueTokenCount, FreqDist) VALUES (?, ?, ?, ?, ?, ?)', 
                (SpotifyID, Sentiment, OriginalCharCount, TokenCount, UniqueTokenCount, FreqDist))
    conn.commit()

def add_to_genremetadata(Genre, OverallTokenCount, GenreUniqueTokenCount, NumSongs, MeanTokenCount, MeanUniqueTokenCount, OverallFreqDist, MeanPopularityOnQuery, MeanAcousticness, MeanDanceability, MeanDuration, MeanEnergy, RatioInMajorKey, MeanSpeechiness, MeanTempo, MeanValence):
    cur.execute('INSERT INTO GenreMetadata (Genre, OverallTokenCount, GenreUniqueTokenCount, NumSongs, MeanTokenCount, MeanUniqueTokenCount, OverallFreqDist, MeanPopularityOnQuery, MeanAcousticness, MeanDanceability, MeanDuration, MeanEnergy, RatioInMajorKey, MeanSpeechiness, MeanTempo, MeanValence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
                (Genre, OverallTokenCount, GenreUniqueTokenCount, NumSongs, MeanTokenCount, MeanUniqueTokenCount, OverallFreqDist, MeanPopularityOnQuery, MeanAcousticness, MeanDanceability, MeanDuration, MeanEnergy, RatioInMajorKey, MeanSpeechiness, MeanTempo, MeanValence))
    conn.commit()

def add_to_overallmetadata(Method, OverallTokenCount, OverallUniqueTokenCount, NumSongs, MeanTokenCount, MeanUniqueTokenCount, OverallFreqDist, MeanPopularityOnQuery, MeanAcousticness, MeanDanceability, MeanDuration, MeanEnergy, RatioInMajorKey, MeanSpeechiness, MeanTempo, MeanValence):
    cur.execute('INSERT INTO OverallMetadata (Method, OverallTokenCount, OverallUniqueTokenCount, NumSongs, MeanTokenCount, MeanUniqueTokenCount, OverallFreqDist, MeanPopularityOnQuery, MeanAcousticness, MeanDanceability, MeanDuration, MeanEnergy, RatioInMajorKey, MeanSpeechiness, MeanTempo, MeanValence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
           (Method, OverallTokenCount, OverallUniqueTokenCount, NumSongs, MeanTokenCount, MeanUniqueTokenCount, OverallFreqDist, MeanPopularityOnQuery, MeanAcousticness, MeanDanceability, MeanDuration, MeanEnergy, RatioInMajorKey, MeanSpeechiness, MeanTempo, MeanValence))
    conn.commit()


# I use a specially formatted dictionary I refer to as "Freq Dists" or frequency distribution
# Each word/token is a dictionary key
# The value associated with a key is how many occurrences have been seen so far
# Input a list of words with potential repeats to get a "Freq Dist" dictionary of the frequency counts
def tokens_to_freqdist(tokens):
    freqdist = {}
    for token in tokens:
        if token not in freqdist.keys():
            freqdist[token]=1
        else:
            freqdist[token]+=1
    return freqdist


# Adds the counts from the "addend" freq dist onto the "base" freq dist
# NOTE: This modifies the base argument. There is no return value
def sum_freqdist(base, addend):
    for token in addend.keys():
        if token not in base.keys():
            base[token]=addend[token]
        else:
            base[token]+=addend[token]


# queries the spotify and genius APIs to put song and lyric data into their respective tables
# if there are 90 genres and you collect 10 per genre, 900 queries will be made
# each time a random genre is picked to query
# this means if a loop errors in the middle it won't affect the proportional representation of each genre
# A LOT of different errors can happen. Any failure of a song and the loop just moves to the next iteration 
# without changing the data table.
# Also note that the current method will detect progressively more repeats (and get slower) as dataset size increases.
# Querying more than one song from spotify at once may help with repeats if a very large set is desired.
def collect_more_songs(num_per_genre):
    G = len(GENRE_SEEDS)
    for g in range(0, G * num_per_genre): 
        try:
            genre_seed = GENRE_SEEDS[np.random.randint(len(GENRE_SEEDS))]
            results = spotify.recommendations(seed_genres=[genre_seed], limit = 1) #20 by default
            for i in range(0, len(results["tracks"])):

                t = results['tracks'][i]
                name = ""
                popularity = ""
                main_artist = ""
                date = ""
                lyrics = ""
                try:
                    print("\n" + t['name'] + "\n(index "+str(g) + " of "+str(G*num_per_genre)+")")
                    name = t['name']
                except:
                    print("Name error")
                    continue
                try:
                    sp_id = t['id']
                except:
                    print("Spotify ID Error")
                    continue
                #print(t.keys())
                #for k in t.keys():
                #  print(t[k])
                if key_in_table(sp_id, "SpotifyID", "SongData"):
                    print("Repeat found")
                    continue
                try:
                    popularity = (t["popularity"])
                    main_artist = (t['artists'][0]['name'])
                    #print("Trying to extract release year from: " + str(t))
                    date = (t['album']['release_date'])
                    if date != None:
                        date = date[:4]
                except Exception as e:
                    print("Other Metadata Error - some metadata will be blank. ",e)
                song = genius.search_song(title=name, artist=main_artist)
                try:
                    hit = song.to_json()
                    hit = json.loads(hit)
                except Exception as e:
                    print("JSON object error:",e)
                    continue
                if "language" not in hit.keys() or "lyrics" not in hit.keys():
                    print("JSON object error type 2")
                    continue
                if hit['language'] != "en":
                    print("Non-English Song: ", hit['language'])
                    continue
                lyrics = hit['lyrics']
                if len(lyrics)>4000:
                    print("Too long to be a song.")
                    continue

                if "Lyrics" in lyrics:
                    lyrics = lyrics.split("Lyrics")[1]
                if "Embed" in lyrics:
                    lyrics = lyrics.split("Embed")[0]
                if "You might also like" in lyrics:
                    lyrics = lyrics.split("You might also like")[0]
                if "Get tickets as low as" in lyrics:
                    lyrics = lyrics.split("Get tickets as low as")[0]
                if "See "+hit['artist']+" Live" in lyrics:
                    lyrics = lyrics.split("See "+hit['artist']+" Live")[0]

                #All tests passed
                #Get audio features
                try:
                    af = spotify.audio_features([sp_id])[0]
                    #print("Audiofeatures: ",af)

                    #Add song to tables
                    add_to_songdata(sp_id, name, main_artist, date, genre_seed, popularity, 
                                    float(af['acousticness']), 
                                    float(af['danceability']), 
                                    int(int(af['duration_ms'])/1000), 
                                    float(af['energy']),
                                    int(af['mode']),
                                    float(af['speechiness']),
                                    int(af['tempo']),
                                    float(af['valence']))

                    #cur.execute('SELECT * FROM SongData')
                    #data = cur.fetchall()
                    #print("Entries in songdata table",len(data))

                    add_to_fulllyricdata(sp_id, int(hit['id']), lyrics)

                    #cur.execute('SELECT * FROM FullLyricData')
                    #data = cur.fetchall()
                    #print("entries in fulllyricdata table: ",len(data))
                except Exception as e:
                    print("error getting audio features: ",e)
                    continue
        except Exception as e:
            global errs #Modify the global scope version of errs, not creating a new one
            errs+=1
            print("Err "+str(errs)+" Likely a timeout: "+str(e))
    print("Collect Songs Loop finished")
            

# Uses the information from SongData and FullLyricData to make ProcessedSongData for all songs that
# don't alreadt have a processed table entry
def process_table_update():
    sid = SentimentIntensityAnalyzer() #Only instantiate the analyzer once per processing cycle
    cur.execute('select SpotifyID, FullLyrics from FullLyricData')
    data = cur.fetchall()
    for d in data:
        #print(d)
        sp_id = d[0]
        if key_in_table(sp_id, "SpotifyID", "ProcessedSongData"):
            #print("Key was already in table.")
            continue
        lyrics = d[1]
        OG_charchount = len(lyrics)

        ##codeblock source: myself in my digital humanities HUM346 project, 2020
        cleaned_tokens = []
        for token in nltk.word_tokenize(lyrics):
            token_lower = token.lower()
            if (token_lower not in string.punctuation) and (token_lower not in stopwords.words('english')):
                #if (token_lower not in "”“" and 
                if (token_lower not in 'like people really even get going'):
                    cleaned_tokens.append(lemmatizer.lemmatize(token_lower))   
        ##end codeblock

        freqdist = tokens_to_freqdist(cleaned_tokens)
        sent = get_sentiment(sid, lyrics)
        add_to_processedsongdata(sp_id, sent, OG_charchount, len(cleaned_tokens), len(freqdist.keys()), str(freqdist))
        print("Done processing:"+str(sp_id))
    print("Process Table Loop finished")


def genre_metadata_update():
    #clear genre metadata table:
    cur.execute('DROP TABLE IF EXISTS GenreMetadata')
    conn.commit()
    cur.execute('CREATE TABLE GenreMetadata (Genre VARCHAR, OverallTokenCount INT, GenreUniqueTokenCount INT, NumSongs INT, MeanTokenCount DOUBLE, MeanUniqueTokenCount DOUBLE, OverallFreqDist VARCHAR, MeanPopularityOnQuery DOUBLE, MeanAcousticness DOUBLE, MeanDanceability DOUBLE, MeanDuration DOUBLE, MeanEnergy DOUBLE, RatioInMajorKey DOUBLE, MeanSpeechiness DOUBLE, MeanTempo DOUBLE, MeanValence DOUBLE)')
    conn.commit()

    #Want to make:
    #OverallTokenCount	OverallUniqueTokenCount	NumSongs	MeanTokenCount	MeanUniqueTokenCount	
    ##OverallFreqDist	MeanPopularityOnQuery	MeanAcousticness	MeanDanceability	
    ##MeanDuration	MeanEnergy	RatioInMajorKey	MeanSpeechiness	MeanTempo	MeanValence

    #We have:
    #PopularityOnQuery	Acousticness	Danceability	Duration	Energy	MajorKey	Speechiness	Tempo	Valence
    #Sentiment	OriginalCharCount	OriginalWordCount	TokenCount	UniqueTokenCount	FreqDist
    for Genre in GENRE_SEEDS:
        #Get all data we need for everything matching Genre
        cur.execute('select SongData.PopularityOnQuery, SongData.Acousticness, SongData.Danceability, SongData.Duration, '+
                    'SongData.Energy, SongData.MajorKey, SongData.Speechiness, SongData.Tempo, SongData.Valence, '+
                    'ProcessedSongData.Sentiment, ProcessedSongData.OriginalCharCount, '+
                    'ProcessedSongData.TokenCount, ProcessedSongData.UniqueTokenCount, ProcessedSongData.FreqDist '+
                    'from SongData inner join ProcessedSongData on SongData.SpotifyID = ProcessedSongData.SpotifyID '+
                    'where SongData.MainGenre ="'+Genre+'"')
        data = cur.fetchall()
        if(len(data)==0):
            print("Suggest Remove Genre "+Genre) #Helped early in data collection to identify non-english or instrumental genres to weed out
            continue
        for i in range(0, len(data)):
            data[i] = list(data[i])
        data = np.array(data)
        data = np.transpose(data) # now, each row is the same data type

        #Helps to keep straight where in the query this data is without having to re-query
        #This should be changed if the order of the columns in the query is changed
        column_key = {
            "Popularity":0,
            "Acousticness":1,
            "Danceability":2,
            "Duration":3,
            "Energy":4,
            "MajorKey":5,
            "Speechiness":6,
            "Tempo":7,
            "Valence":8,
            "Sentiment":9,
            "OriginalCharCount":10,
            "TokenCount":11,
            "UniqueTokenCount":12,
            "FreqDist":13
            }



        OverallTokenCount = np.sum(np.array(data[column_key["UniqueTokenCount"]], dtype='float64')) #check these after WC deletion

        OverallFreqDist = {}
        for f in data[column_key["FreqDist"]]:
            #print(f)
            sum_freqdist(OverallFreqDist, ast.literal_eval(f))
        OverallUniqueTokenCount = len(OverallFreqDist.keys())
        NumSongs = len(data[column_key["Popularity"]]) #all cols should be the same length
        MeanTokenCount = OverallTokenCount / NumSongs
        MeanUniqueTokenCount = np.mean(np.array(data[column_key["UniqueTokenCount"]], dtype='float64'))
        MeanPopularityOnQuery = np.mean(np.array(data[column_key["Popularity"]], dtype='float64'))
        MeanAcousticness = np.mean(np.array(data[column_key["Acousticness"]], dtype='float64'))
        MeanDanceability = np.mean(np.array(data[column_key["Danceability"]], dtype='float64'))
        MeanDuration = np.mean(np.array(data[column_key["Duration"]], dtype='float64'))
        MeanEnergy = np.mean(np.array(data[column_key["Energy"]], dtype='float64'))
        RatioInMajorKey = np.mean(np.array(data[column_key["MajorKey"]], dtype='float64'))
        MeanSpeechiness = np.mean(np.array(data[column_key["Speechiness"]], dtype='float64'))
        MeanTempo = np.mean(np.array(data[column_key["Tempo"]], dtype='float64'))
        MeanValence = np.mean(np.array(data[column_key["Valence"]], dtype='float64'))
        add_to_genremetadata(Genre, OverallTokenCount, OverallUniqueTokenCount, NumSongs, MeanTokenCount, MeanUniqueTokenCount, str(OverallFreqDist), MeanPopularityOnQuery, MeanAcousticness, MeanDanceability, MeanDuration, MeanEnergy, RatioInMajorKey, MeanSpeechiness, MeanTempo, MeanValence)
        print("Done with "+Genre)
    print("Genre Metadata Update Loop Finished")

    
    

def overall_metadata_update():
    cur.execute('DROP TABLE IF EXISTS OverallMetadata')
    conn.commit()
    cur.execute('CREATE TABLE OverallMetadata (Method VARCHAR, OverallTokenCount INT, OverallUniqueTokenCount INT, NumSongs INT, MeanTokenCount DOUBLE, MeanUniqueTokenCount DOUBLE, OverallFreqDist VARCHAR, MeanPopularityOnQuery DOUBLE, MeanAcousticness DOUBLE, MeanDanceability DOUBLE, MeanDuration DOUBLE, MeanEnergy DOUBLE, RatioInMajorKey DOUBLE, MeanSpeechiness DOUBLE, MeanTempo DOUBLE, MeanValence DOUBLE)')
    conn.commit()

    cur.execute('select * from GenreMetadata')
    data = cur.fetchall()
    Method="Recalculated" #Legacy from when the OverallMetadata table had multiple rows involving incremental calculations
    #Currently this table only ever has one row, but it works 
    NumSongs = 0
    for row in data:
        NumSongs += row[3]
    OverallTokenCount=0
    OverallFreqDist = {}
    MeanTokenCount=0
    MeanUniqueTokenCount=0
    MeanPopularityOnQuery=0
    MeanAcousticness=0
    MeanDanceability=0
    MeanDuration=0
    MeanEnergy=0
    RatioInMajorKey=0
    MeanSpeechiness=0
    MeanTempo=0
    MeanValence=0
    #Ends with an average weighted by the genres'
    for row in data:
        #row indices left as plain numbers here because the two metadata tables have extremely similar sets of columns

        GenreWeight = float(row[3])/NumSongs

        OverallTokenCount += row[1]
        MeanTokenCount += row[4] * GenreWeight
        MeanUniqueTokenCount += row[5] * GenreWeight
        MeanPopularityOnQuery += row[7] * GenreWeight
        MeanAcousticness += row[8] * GenreWeight
        MeanDanceability += row[9] * GenreWeight
        MeanDuration += row[10] * GenreWeight
        MeanEnergy += row[11] * GenreWeight
        RatioInMajorKey += row[12] * GenreWeight
        MeanSpeechiness += row[13] * GenreWeight
        MeanTempo += row[14] * GenreWeight
        MeanValence += row[15] * GenreWeight
        sum_freqdist(OverallFreqDist, ast.literal_eval(row[6])) 

    OverallUniqueTokenCount = len(OverallFreqDist.keys())
    add_to_overallmetadata(Method, OverallTokenCount, OverallUniqueTokenCount, NumSongs, MeanTokenCount, MeanUniqueTokenCount, str(OverallFreqDist), MeanPopularityOnQuery, MeanAcousticness, MeanDanceability, MeanDuration, MeanEnergy, RatioInMajorKey, MeanSpeechiness, MeanTempo, MeanValence)
    print("Finished Overall Metadata")

# return the compound sentiment (-1 very negative, +1 very positive)
# sid is the analyzer object and essentially should only get constructed once and not reused
def get_sentiment(sid, lyrics):
    ss = sid.polarity_scores(lyrics)
    print(ss)
    return ss['compound']


###### MAIN PROGRAM
#Somewhat like a jupyter notebook, comment and uncomment or change these lines
#to run only what you want in the file.
#clear_tables()

collect_more_songs(1)
process_table_update()
genre_metadata_update()
overall_metadata_update()

print("Done. timeout errs: "+str(errs))

print("Generating python-formatted dictionary variable to bypass table queries...")

import FetchDataModule #The other py file with code originally intended to be in the server backend
#Produces a python-formatted dictionary with all possible results combinations
# intended to be copied and pasted into the anvil text.
#See other file comments for details
print("\n\n\n\n\n\n\n\n\n\n\n"+FetchDataModule.nested_dictionary_cache_text())

#Close opened connections
FetchDataModule.my_conn.close()
conn.close()

print("Program Complete")



