from django.shortcuts import render
import json
import youtube_dl
import librosa
import os
from pytube import YouTube
from pydub import AudioSegment  
from tensorflow.keras.models import load_model
import numpy as np
from collections import Counter
from django.core.files.storage import default_storage
# Create your views here.

# 
genre_dict = {
    0: 'Blues',
    1: 'Classical',
    2: 'Country',
    3: 'Disco', 
    4: 'Hiphop', 
    5: 'Jazz',
    6: 'Metal', 
    7: 'Pop',
    8: 'Reggae', 
    9: 'Rock', 
}

def home(request):
    if request.method == 'POST':
        youtube_url = request.POST.get('youtube_url', None)
        audio = request.POST.get('audio_file', None)
        if not youtube_url and not audio:
            return render(request, 'index.html', {'genre_dict' : genre_dict})
        if not youtube_url:
            audio_file = request.FILES['audio_file']
            
            if not request.POST.get('audio_file', None):
                return render(request, 'index.html', {'genre_dict' : genre_dict})
            # process the audio file
            # result = process_audio(audio_file)
            result = process_audio(audio_file, type = 'audio')
        else: 
            result = process_audio(youtube_url, type = 'youtube')
            # result = process_youtube_url(youtube_url)
        return render(request, 'index.html', {'genre_dict': genre_dict, 'result' : result} )
    else:
        return render(request, 'index.html', {'genre_dict' : genre_dict})



# fetch the audio from youtube video and convert it to wav format
def process_audio(file_link, type = 'youtube'):
    if type == 'youtube':
        yt = YouTube(file_link)
        video = yt.streams.filter(only_audio=True).first()
        destino = "media"
        out_file = video.download(output_path=destino)
        audio = AudioSegment.from_file(out_file)
        base, ext = os.path.splitext(out_file)
    else:
        default_storage.save(file_link.name, file_link)
        out_file = "media/"+file_link.name
        audio = AudioSegment.from_file(out_file)
        base, ext = os.path.splitext(out_file)
        
    new_file = base + '.wav'
    audio.export(new_file, format='wav')

    mfccs = generate_mfccs(new_file)
    # return mfccs
    result = genre_predictions(mfccs)

    if type == 'youtube':
        default_storage.delete(os.path.basename(base)+".wav")
    default_storage.delete(os.path.basename(out_file))
    return result


# Do the same preprocessing steps as during the model training
def generate_mfccs(audio_file):
    
    AUDIO_MAX_LENGTH = 30
    AUDIO_SLICE_DURATION = 3
    # audio slices 
    TOTAL_SLICES = AUDIO_MAX_LENGTH // AUDIO_SLICE_DURATION

    # sampling rate
    sr = 22050
    # To make sure, all files have sample amount of samples, choose the duration less than audio file
    TOTAL_SAMPLES = 29 * sr
    SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / TOTAL_SLICES)
    data_dict = {
        "mfccs" : []
    }
    audio, sample_rate = librosa.load(audio_file, duration=AUDIO_MAX_LENGTH)
    for s in range(TOTAL_SLICES):
        start_sample = SAMPLES_PER_SLICE * s
        end_sample = start_sample + SAMPLES_PER_SLICE
        mfcc = librosa.feature.mfcc(y=audio[start_sample:end_sample], sr=sr, n_mfcc=13)
        if mfcc is not None:
            mfcc = mfcc.T
            # append the features and genre type to make dataset

            data_dict["mfccs"].append(mfcc.tolist())
            
    X = np.array(data_dict["mfccs"])
    X = X[..., np.newaxis]
    return X


def genre_predictions(X):
    model1 = load_model('models/cnn_model-2.h5')
    model2 = load_model('models/tuned_cnn_model.h5')
    model3 = load_model('models/rnn_model.h5')
    prediction = model1.predict(X)

    genre_list = list()
    # As the mfcc of audio files are generated for each 3 second slice, 
    # looping for all the values and return the genre that is repeated most
    for index, value in enumerate(prediction):
        predicted_genre = np.argmax(prediction[index])
        genre_list.append(genre_dict[predicted_genre])
    print(genre_list)
    counts = Counter(genre_list)
    most_repeated_genre = max(counts, key=counts.get)

    print(f"Predicted Genre: {most_repeated_genre}")
    return most_repeated_genre
    