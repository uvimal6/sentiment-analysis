# - *- coding: utf- 8 - *-
import streamlit as st
import pandas as pd
import hashlib
import sqlite3
import numpy as np
import speech_recognition as sr
import speech_recognition as sp
import paralleldots
from textblob import TextBlob
import contextlib
import wave
import math
import re
import scipy.io.wavfile
from scipy.io import wavfile
import matplotlib.pyplot as plt
import moviepy.editor
import os,shutil
from googletrans import Translator
import librosa
mysp=__import__("my-voice-analysis")
import librosa.display
import IPython.display as ipd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#from ibm_watson import ToneAnalyzerV3
from pprint import pprint
#from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from pydub import AudioSegment
# Security
#passlib,hashlib,bcrypt,scrypt
import pyaudio
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier
import soundfile
import glob
from sklearn.model_selection import train_test_split
from pdf_mail import sendpdf
from pydub import AudioSegment
from pydub.silence import split_on_silence
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
document=[]


def envelope( ):
    mask=[]
    threshold=0.0005
    file="sample.wav"
    y , rate = librosa.load(file, sr=16000)
    
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    wavfile.write(r'C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream\\'+str(file), rate=rate,data=y[mask])   
    st.info("Noise Reduction Done")    
    return mask

def drawMyRuler(pdf):
    pdf.drawString(100,810, 'x100')
    pdf.drawString(200,810, 'x200')
    pdf.drawString(300,810, 'x300')
    pdf.drawString(400,810, 'x400')
    pdf.drawString(500,810, 'x500')

    pdf.drawString(10,100, 'y100')
    pdf.drawString(10,200, 'y200')
    pdf.drawString(10,300, 'y300')
    pdf.drawString(10,400, 'y400')
    pdf.drawString(10,500, 'y500')
    pdf.drawString(10,600, 'y600')
    pdf.drawString(10,700, 'y700')
    pdf.drawString(10,800, 'y800')




#page_bg_img = '''
#<style>
#body {
#background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
#background-size: cover;
#}
#</style>
#'''

#st.markdown(page_bg_img, unsafe_allow_html=True)

model = pickle.load(open("mlp_classifier.model", "rb"))
# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    #"03": "happy",
    "03": "sad",
    "04": "angry",
    #"06": "fearful",
    "05": "disgust",
    "06": "surprised"
}

# we allow only these emotions
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    #"happy",
    "disgust"
}





# load the saved model (after training)










def make_hashes(password):
	return password

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management

conn = sqlite3.connect('data.db')
c = conn.cursor()
apikey = 'OoDWxD9rNgjw-79bjhr1MsVGcuqa8kU9OeiJbaJQlaWI'
url = 'https://api.us-south.tone-analyzer.watson.cloud.ibm.com/instances/998709a9-2c05-4199-b160-b9cf027a7425'
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data
def sppech():
        r = sr.Recognizer()
        audio = 's.wav'
        with sr.AudioFile(audio) as source:
            audio = r.record(source)
            st.write('Done!')
        try:
            text = r.recognize_google(audio,language='en-IN')
            st.write(text)
        except Exception as e:
            print(e)
        with open('text.txt', 'w') as f:
            word = text
            f.write(word + '\n')
def audio():
        audio = 'sss.wav'
        paralleldots.set_api_key("8DhrXaaW5mRir7398Ut0hmvYElXfREMtpF4ovagK0wY")
        response = paralleldots.emotion(audio)
        st.write(response)
def translator():
	q=st.text_area("enter the text")
	if st.button("Translate"):
		t = Translator()
		translation = t.translate(q)
		st.success(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
def translation(c):
  translator = Translator()
  ch=translator.translate(c).text
  lang=translator.translate(c).src
  st.write(ch,lang)
def plot_spectrogram():
    """Compute power spectrogram with Short-Time Fourier Transform and plot result."""
    violin_sound_file = r"sample.wav"
    my_expander3 = st.beta_expander("PLOT SPECTOGRAM",expanded=False)
    violin_c4, _ = librosa.load(violin_sound_file)
    ipd.Audio(violin_sound_file)
    spectrogram = librosa.amplitude_to_db(librosa.stft(violin_c4))
    plt.figure(figsize=(20, 15))
    librosa.display.specshow(spectrogram, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-frequency power spectrogram for audio")
    plt.xlabel("Time")
    plt.savefig('LOGFPS.png')
    my_expander3.image('LOGFPS.png')
    X = np.fft.fft(violin_c4)
    X_mag = np.absolute(X)
    f = np.linspace(0, _, len(X_mag))
    plt.figure(figsize=(18, 10))
    plt.plot(f, X_mag) # magnitude spectrum
    plt.xlabel('Frequency (Hz)')
    plt.title("Frequency")
    plt.savefig('Freq.png')
    my_expander3.image('Freq.png')
    my_expander3.write("Length of the Frequency")
    my_expander3.write(len(violin_c4))
def wit():
	#Loading Audio Files
	mala_file = r"sample.wav"
	my_expander4 = st.beta_expander("ZCR AND RMSE",expanded=False)
	ipd.Audio(mala_file)
	# load audio files with librosa
	mala, sr = librosa.load(mala_file)
	#Root-mean-squared energy with Librosa
	FRAME_SIZE = 1024
	HOP_LENGTH = 512
	rms_mala = librosa.feature.rms(mala, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
	#Visualise RMSE + waveform
	frames = range(len(rms_mala))
	t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
	# rms energy is graphed in red
	plt.figure(figsize=(15, 17))
	plt.subplot(3, 1, 2)
	librosa.display.waveplot(mala, alpha=0.5)
	plt.plot(t, rms_mala, color="r")
	plt.ylim((-1, 1))
	plt.title("RMSE + WAVEFORM")
	plt.savefig('RMSEWAV.png')
	my_expander4.image('RMSEWAV.png')
	#Zero-crossing rate with Librosa
	zcr_mala = librosa.feature.zero_crossing_rate(mala, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
	my_expander4.write("size of ZCR")
	my_expander4.write(zcr_mala.size)
	#Visualise zero-crossing rate with Librosa
	plt.figure(figsize=(15, 10))
	plt.plot(t, zcr_mala, color="r")
	plt.title("ZCR")
	plt.ylim(0, 1)
	plt.savefig('ZCR.png')
	my_expander4.image('ZCR.png')
	#ZCR: Voice vs Noise
	voice_file = r"sample.wav"
	noise_file = r"sample.wav"

	ipd.Audio(voice_file)

	ipd.Audio(noise_file)

	# load audio files
	voice, _ = librosa.load(voice_file, duration=15)
	noise, _ = librosa.load(noise_file, duration=15)

	# get ZCR
	zcr_voice = librosa.feature.zero_crossing_rate(voice, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
	zcr_noise = librosa.feature.zero_crossing_rate(noise, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

	frames = range(len(zcr_voice))
	t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
	frames1 = range(len(zcr_noise))
	t1 = librosa.frames_to_time(frames1, hop_length=HOP_LENGTH)

	plt.figure(figsize=(15, 10))

	plt.plot(t, zcr_voice, color="r")
	plt.plot(t1, zcr_noise, color="y")
	plt.title("ZCR VOICE AND NOISE")
	plt.ylim(0, 1)
	plt.savefig('ZCRVAN.png')
	my_expander4.image('ZCRVAN.png')


def audio1():
	p="sample" # Audio File title
	my_expander6 = st.beta_expander("FEATURES",expanded=False)
	c=r"C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream" # Path to the Audio_File directory (Python 3.7)
	my_expander6.write ("number_ of_syllables=")
	my_expander6.write (mysp.myspsyl(p,c))
	my_expander6.write ("number_of_pauses=")
	my_expander6.write (mysp.mysppaus(p,c))
	my_expander6.write ("rate_of_speech=")
	my_expander6.write (mysp.myspsr(p,c))
	my_expander6.write ("articulation_rate=")
	my_expander6.write (mysp.myspatc(p,c))
	my_expander6.write ("speaking_duration=")
	my_expander6.write (mysp.myspst(p,c))
	my_expander6.write ("original_duration=")
	my_expander6.write (mysp.myspod(p,c))
	my_expander6.write ("balance=")
	my_expander6.write (mysp.myspbala(p,c))
	my_expander6.write ("f0_mean=")
	my_expander6.write (mysp.myspf0mean(p,c))
	my_expander6.write ("f0_SD=")
	my_expander6.write (mysp.myspf0sd(p,c))
	my_expander6.write ("f0_MD=")
	my_expander6.write (mysp.myspf0med(p,c))
	my_expander6.write ("f0_min=")
	my_expander6.write (mysp.myspf0min(p,c))
	my_expander6.write ("f0_max=")
	my_expander6.write (mysp.myspf0max(p,c))
	my_expander6.write ("f0_quan25=")
	my_expander6.write (mysp.myspf0q25(p,c))
	my_expander6.write ("f0_quan75=")
	my_expander6.write (mysp.myspf0q75(p,c))
	my_expander6.write (mysp.mysptotal(p,c))
	my_expander6.write ("Pronunciation_posteriori_probability_score_percentage= :%.2f"% (mysp.mysppron(p,c)))
	
def get_large_audio_transcription():
	menu = ["---------------------------------------------------------SELECT THE LANGUAGE----------------------------------------------------------","en","hi","ml","ta","tl"]
	choice = st.selectbox("languages",menu)
	if choice !="---------------------------------------------------------SELECT THE LANGUAGE----------------------------------------------------------":
		video = st.file_uploader("Select file from your directory")
		if video is not None:
			if video.type=="video/mp4":
				my_expander = st.beta_expander("VIDEO CONVERSION",expanded=False)
				my_expander1 = st.beta_expander("VIDEO ANALYSIS",expanded=False)
				file_details = {"FileName":video.name,"FileType":video.type,"FileSize":video.size}
				my_expander.write(file_details)
				a=video.read()
				my_expander.video(video, format='video/mp4')
				video1 = moviepy.editor.VideoFileClip(video.name)
				audio = video1.audio
				audio.write_audiofile(r"C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream\sample.wav")
				my_expander.audio("sample.wav", format='audio/wav')
				rate,data=scipy.io.wavfile.read('sample.wav')
				my_expander1.write(rate)
				plt.plot(data)
				plt.savefig('graph.png')
				my_expander1.image('graph.png')
				envelope();
				path="sample.wav"
				sound = AudioSegment.from_wav(path)  
				chunks = split_on_silence(sound,
			        # experiment with this value for your target audio file
			        min_silence_len = 500,
			        # adjust this per requirement
			        silence_thresh = sound.dBFS-10,
			        # keep the silence for 1 second, adjustable as well
			        keep_silence=500,
			    )
				folder_name = "audio-chunks"
				if not os.path.isdir(folder_name):
					os.mkdir(folder_name)
				whole_text = "" 
				for i, audio_chunk in enumerate(chunks, start=1):
					# export audio chunk and save it in
					# the `folder_name` directory.
					chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
					audio_chunk.export(chunk_filename, format="wav")
					# recognize the chunk
					r = sp.Recognizer()
					with sp.AudioFile(chunk_filename) as source:
						audio_listened = r.record(source)
						# try converting it to text
						try:
							textw = r.recognize_google(audio_listened,language=choice)
							translator = Translator()
							ch=translator.translate(textw).text
							#lang=translator.translate(c).src 
						except sr.UnknownValueError as e:
							st.write("Error:", str(e))
						else:
							text = f"{ch.capitalize()}. "
							st.write(chunk_filename, ":", ch)
							whole_text += ch
				# return the text for all chunks detected
				st.write(whole_text)
				fl=open("try.txt","w")
				fl.write(whole_text)
				st.write(whole_text)
				fl=open("try.txt","w")
				fl.write(whole_text)
				fl.close()
				text = open("try.txt","rb")
				addParagraph();
				qw=TextBlob(whole_text)
				we=qw.sentiment.polarity
				if we==0:
					st.write("This is a neutral message")
					senti="neutral"
					img(senti);
					st.image("neutral.png",width=180)
				elif we>0:
					st.subheader("This is a positive message")
					senti="positive"
					img(senti);
					st.image("positive.png",width=180)
				else:
					st.write("This is a negative message")
					senti="negative"
					img(senti)
					st.image("negative.png",width=180)
				return 1


			elif video.type=="audio/wav":
				audio=video
				my_expander = st.beta_expander("AUDIO ",expanded=False)
				my_expander1 = st.beta_expander("AUDIO ANALYSIS",expanded=False)
				file_details = {"FileName":audio.name,"FileType":audio.type,"FileSize":audio.size}
				my_expander.write(file_details)
				a=audio.read()
				my_expander.audio(audio, format='audio/wav')
				rate,data=scipy.io.wavfile.read(audio)
				my_expander1.write(rate)
				plt.plot(data)
				plt.savefig('graph.png')
				my_expander1.image('graph.png')
				file_var = AudioSegment.from_ogg(video.name) 
				file_var.export('sample.wav', format='wav')
				envelope();
				path="sample.wav"
				sound = AudioSegment.from_wav(path)  
				chunks = split_on_silence(sound,
			        # experiment with this value for your target audio file
			        min_silence_len = 500,
			        # adjust this per requirement
			        silence_thresh = sound.dBFS-14,
			        # keep the silence for 1 second, adjustable as well
			        keep_silence=500,
			    )
				folder_name = "audio-chunks"
				if not os.path.isdir(folder_name):
					os.mkdir(folder_name)
				whole_text = "" 
				for i, audio_chunk in enumerate(chunks, start=1):
					# export audio chunk and save it in
					# the `folder_name` directory.
					chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
					audio_chunk.export(chunk_filename, format="wav")
					# recognize the chunk
					r = sp.Recognizer()
					with sp.AudioFile(chunk_filename) as source:
						audio_listened = r.record(source)
						# try converting it to text
						try:
							text = r.recognize_google(audio_listened,language=choice)
							translator = Translator()
							ch=translator.translate(text).text
							#lang=translator.translate(c).src 
						except sr.UnknownValueError as e:
							print("Error:", str(e))
						else:
							text = f"{ch.capitalize()}. "
							st.write(chunk_filename, ":",text)
							whole_text += ch
				st.write(whole_text)
				fl=open("try.txt","w")
				fl.write(whole_text)
				fl.close()
				text = open("try.txt","rb")
				addParagraph();
				qw=TextBlob(whole_text)
				we=qw.sentiment.polarity
				if we==0:
					st.write("This is a neutral message")
					senti="neutral"
					img(senti);
					st.image("neutral.png",width=180)
				elif we>0:
					st.subheader("This is a positive message")
					senti="positive"
					img(senti);
					st.image("positive.png",width=180)
				else:
					st.write("This is a negative message")
					senti="negative"
					img(senti)
					st.image("negative.png",width=180)
				shutil.rmtree("audio-chunks")
				os.remove("sample.wav")
				return 1


def img(senti):
    document.append(Paragraph('Emoji:',ParagraphStyle(name='Name',fontFamily='Times New Roman',fontSize=25,alignment=TA_JUSTIFY)))
    document.append(Spacer(1,50))
    if senti=="positive":
    	document.append(Image('positive.png',2.2*inch,2.2*inch))
    	document.append(Spacer(1,50))
    elif senti=="negative":
    	document.append(Image('negative.png',2.2*inch,2.2*inch))
    	document.append(Spacer(1,50))
    else:
    	document.append(Image('neutral.png',2.2*inch,2.2*inch))
    	document.append(Spacer(1,50))
    pdf=SimpleDocTemplate('report.pdf',pagesize=letter,rightMargin=12,leftMargin=12,topmargin=0,bottomMargin=0).build(document)	

def addTitle(doc):
    doc.append(Spacer(1,6))
    doc.append(Paragraph('MULTI-MODAL SENTIMENTAL ANALYSIS',ParagraphStyle(name='Name',fontFamily='Times New Roman',fontSize=25,alignment=TA_CENTER)))
    doc.append(Spacer(1,50))
    
    return doc
def addParagraph():
	addTitle(document)
	document.append(Paragraph('Paragraph :',ParagraphStyle(name='Name',fontFamily='Times New Roman',fontSize=23,alignment=TA_JUSTIFY)))
	document.append(Spacer(1,30))
	with open('try.txt') as txt:
		#document.append(Paragraph(l,ParagraphStyle(name='Name',fontFamily='Times New Roman',fontSize=14,alignment=TA_JUSTIFY)))
		for line in txt.read().split('\n'):
			st.write(line)
			document.append(Paragraph(line,ParagraphStyle(name='Name',fontFamily='Times New Roman',fontSize=14,alignment=TA_JUSTIFY)))
			document.append(Spacer(1,20))
	        
	document.append(Spacer(1,50))
	#SimpleDocTemplate('report.pdf',pagesize=letter,rightMargin=12,leftMargin=12,topmargin=0,bottomMargin=0).build(document)



def mel(audio):
	        scale_file=audio
	        my_expander2 = st.beta_expander("MEL FREQUENCY",expanded=False)
	        ipd.Audio(scale_file)
	        scale, sr = librosa.load(scale_file)
	        filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
	        my_expander2.write(filter_banks.shape)
	        plt.figure(figsize=(25, 10))
	        librosa.display.specshow(filter_banks,sr=sr,x_axis="linear") 
	        plt.colorbar(format="%+2.f")
	        plt.savefig('123.png')
	        my_expander2.image('123.png')
	        mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
	        my_expander2.write(mel_spectrogram.shape)
	        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
	        my_expander2.write(log_mel_spectrogram.shape)
	        plt.figure(figsize=(25, 10))
	        librosa.display.specshow(log_mel_spectrogram,x_axis="time",y_axis="mel",sr=sr)
	        plt.colorbar(format="%+2.f")
	        plt.savefig('1234.png')
	        my_expander2.image('1234.png')

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


def text():
	qq=st.text_area("enter the text")
	if st.button("ANALYSIS"):
		t = Translator()
		translation = t.translate(qq)
		text=translation.text
		st.write(text)
		qw=TextBlob(text)
		we=qw.sentiment.polarity
		if we==0:
			st.write("This is a neutral message")
			st.image("neutral.png",width=180)
		elif we>0:
			st.write("This is a positive message")
			st.image("positive.png",width=180)
		else:
			st.write("This is a negative message")
			st.image("negative.png",width=180)
def video():
	menu = ["---------------------------------------SELECT THE LANGUAGE-----------------------------------","en","hi","ml","ta","tl"]
	choice = st.selectbox("",menu)
	if choice !="---------------------------------------SELECT THE LANGUAGE-----------------------------------":
		video = st.file_uploader("Select file from your directory")
		if video is not None:
			if video.type=="video/mp4":
				my_expander = st.beta_expander("VIDEO CONVERSION",expanded=False)
				my_expander1 = st.beta_expander("VIDEO ANALYSIS",expanded=False)
				file_details = {"FileName":video.name,"FileType":video.type,"FileSize":video.size}
				my_expander.write(file_details)
				a=video.read()
				my_expander.video(video, format='video/mp4')
				video1 = moviepy.editor.VideoFileClip(video.name)
				audio = video1.audio
				audio.write_audiofile(r"C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream\sample.wav")
				my_expander.audio("sample.wav", format='audio/wav')
				audio1();
				audio='sample.wav'
				rate,data=scipy.io.wavfile.read('sample.wav')
				my_expander.write("RATE OF AUDIO")
				my_expander1.write(rate)
				my_expander1.write(data)
				plt.plot(data)
				plt.savefig('graph2.png')
				my_expander1.image('graph2.png')
				mel(audio);
				plot_spectrogram();
				wit();
				pas=(mysp.myspgend("sample",r"C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream"))
				if pas[2]==0:
					st.image("male.png",width=230)
					if pas[3]==1:
						st.write("a male, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % pas[0] ,pas[1])
					elif pas[3]==2:
						
						st.write("a male, mood of speech: Reading, p-value/sample size= :%.2f" % pas[0], pas[1])
					else:
						st.write("a male, mood of speech: speaking passionately, p-value/sample size= :%.2f" % pas[0], pas[1])
				else:
					st.image("fe.png",width=230)
					if pas[3]==1:
						st.write("a female, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % pas[0] ,pas[1])
					elif pas[3]==2:
						st.write("a female, mood of speech: Reading, p-value/sample size= :%.2f" % pas[0], pas[1])
					else:
						st.write("a female, mood of speech: speaking passionately, p-value/sample size= :%.2f" % pas[0], pas[1])	

			else:
				audio=video
				my_expander = st.beta_expander("AUDIO ",expanded=False)
				my_expander1 = st.beta_expander("AUDIO ANALYSIS",expanded=False)
				file_details = {"FileName":audio.name,"FileType":audio.type,"FileSize":audio.size}
				my_expander.write(file_details)
				a=audio.read()
				my_expander.audio(audio, format='audio/wav')
				rate,data=scipy.io.wavfile.read(audio)
				my_expander1.write(rate)
				my_expander1.write(data)
				plt.plot(data)
				plt.savefig('graph.png')				
				my_expander1.image('graph.png')
				my_expander12 = st.beta_expander("MEL FREQUENCY",expanded=False)
				scale_file=video
				ipd.Audio(scale_file.name)
				scale, sr = librosa.load(scale_file)
				filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
				my_expander12.write(filter_banks.shape)
				plt.figure(figsize=(25, 10))
				librosa.display.specshow(filter_banks,sr=sr,x_axis="linear")
				plt.colorbar(format="%+2.f")
				plt.savefig('123.png')
				my_expander12.image('123.png')
				mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
				my_expander12.write(mel_spectrogram.shape)
				log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
				my_expander12.write(log_mel_spectrogram.shape)
				plt.figure(figsize=(25, 10))
				librosa.display.specshow(log_mel_spectrogram,x_axis="time",y_axis="mel",sr=sr)
				plt.colorbar(format="%+2.f")
				plt.savefig('1234.png')
				my_expander12.image('1234.png')

				file_var = AudioSegment.from_ogg(video.name) 
				file_var.export('sample.wav', format='wav')
				audio1();
				plot_spectrogram();
				wit();
				filename = "sample.wav"
				features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
				result = model.predict(features)[0]
				st.write(result)	
				if result=="happy":
				   st.image('neutral.png',width=230)
				else:
				   st.image('happy.png')					
				pas=(mysp.myspgend("sample",r"C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream"))
				if pas[2]==0:
					st.image("male.png",width=230)
					if pas[3]==1:
						st.write("a male, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % pas[0] ,pas[1])
					elif pas[3]==2:
						st.write("a male, mood of speech: Reading, p-value/sample size= :%.2f" % pas[0], pas[1])
					else:
						st.write("a male, mood of speech: speaking passionately, p-value/sample size= :%.2f" % pas[0], pas[1])
				else:
					st.image("fe.png",width=230)
					if pas[3]==1:
						st.write("a female, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % pas[0] ,pas[1])
					elif pas[3]==2:
						st.write("a female, mood of speech: Reading, p-value/sample size= :%.2f" % pas[0], pas[1])
					else:
						st.write("a female, mood of speech: speaking passionately, p-value/sample size= :%.2f" % pas[0], pas[1])	

def main():
	menu = ["Login","SignUp","Admin login"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home")
	elif choice == "Login":
		#st.subheader("Login Section")
		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)
			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:

				st.success("YOU LOGGED IN AS {}".format(username))
				st.markdown('<h1><center style= color:#00bfff;>WELCOME</center></h1>', unsafe_allow_html=True)
				#st.title("<h1>WELCOME</h1>")
				st.write("Build with Streamlit")
				activites=["--------SELECT THE OPTION--------","VIDEO|AUDIO","NLP","TEXT ANALYSIS","TRANSLATION","ABOUT"]
				choices=st.sidebar.selectbox("Select Activities",activites)
				if choices=="NLP":
					st.write("NLP")
					#st.write("\nFull text:", get_large_audio_transcription())
					sdf=get_large_audio_transcription()
					if sdf==1:
						k=sendpdf("newtechcreator6@gmail.com",#sender mailid
				          "newtechcreator6@gmail.com",#reciever mailid
				          "projectmailid",#sender pass
				          "Report of Analysis",#subject of the message
				          "Message from Strealit app",#body of the message
				          "report",#filename
				          "C:/Users/uvima/Desktop/Streamlit app/Sentimental Analysis final/Stream")#filepath
						k.email_send()
						st.info("Report has been successfully sent to your email")
				if choices=="ABOUT":
					      st.write("This is My final year project")
				if choices=="TEXT ANALYSIS":
			    	       st.write("this is Text Analysis")
			    	       text();
				if choices=="VIDEO|AUDIO":
					      video();
				if choices=="TRANSLATION":	
					  translator();			  
			else:
				st.warning("INCORRECT Username|Password")
	elif choice=="Admin login":
			username = st.sidebar.text_input("User Name")
			password = st.sidebar.text_input("Password",type='password')
			if st.sidebar.checkbox("Login"):
				if username=="ad" and password=="ad":
					st.success(" LOGGED IN AS ADMIN ")
					task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
					if task == "Add Post":
						st.subheader("Add Your Post")

					elif task == "Analytics":
						st.subheader("Analytics")
					elif task == "Profiles":
						st.subheader("User Profiles")
						user_result = view_all_users()
						clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
						st.dataframe(clean_db)
				else:
					st.warning("INCORRECT Username|Password")


	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")



if __name__ == '__main__':
	main()
