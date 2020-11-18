# importing libraries
from pyAudioAnalysis import audioBasicIO , audioAnalysis , audioSegmentation , audioTrainTest
import matplotlib.pyplot as plt
import pandas as pd
import speech_recognition as sr
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# audio file path (english audio)
filename = 'Gravação.wav'

# reading audio frequency
[fs , x] = audioBasicIO.read_audio_file(filename)

# ploting the audio frequency
plt.plot(x)
plt.grid()

rec = sr.Recognizer()

with sr.AudioFile(filename) as source:
    # listen for the data (load audio to memory)
    audio_data = rec.record(source)
    # recognize (convert from speech to text)
    text = rec.recognize_google(audio_data , language='en-US') # to change language: https://cloud.google.com/speech-to-text/docs/languages
    print(text)

stopwords = set(STOPWORDS)
stopwords.update(['the', 'you', 'i', 'me', 'my', 'he', 'she','your','and','we','are'])

