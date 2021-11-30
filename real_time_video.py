from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from playsound import playsound

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
            "neutral"]

Face_emotion = ''
# starting video streaming
cv2.namedWindow('Emotion_classifier')
camera = cv2.VideoCapture(0) # change number to detect any other camera
#camera = cv2.VideoCapture('') # for video file
while True:

    # emotion_percentage = {"angry":0,"disgust":0,"scared":0, "happy":0,
    #                 "sad":0, "surprised":0, "neutral":0}

    frame = camera.read()[1] # this is for live video
    # frame = cv2.imread(r'/home/sid/Desktop/images.jpeg') # this is for image
    # print(frame)
    #reading the frame
    # frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img',gray)
    faces = face_detection.detectMultiScale(gray,flags=cv2.CASCADE_SCALE_IMAGE)#,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    # print(len(faces))
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    Total_no_of_faces = len(faces)

    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))#[0]
        for face in faces:
            (fX, fY, fW, fH) = face

            # print(len(faces))
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            # print('roi',roi.shape)
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # print('roi',roi.shape)
            
            
            preds = emotion_classifier.predict(roi)[0]
            # print(preds)
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            if label == "angry":
                Face_emotion = "sad"

            if label == "happy":
                Face_emotion = "happy"
            
            if label == "sad":
                Face_emotion = "sad"
            
            if label == "disgust":
                Face_emotion = "sad"
            
            if label == "scared":
                Face_emotion = "sad"
            
            if label == "surprised":
                Face_emotion = "happy"
            
            if label == "neutral":
                Face_emotion = "happy"


     
            # for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                
            cv2.putText(frameClone, Face_emotion, (fX, fY - 10),#for all emotion change to label
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 255, 255), 2)
            cv2.putText(frameClone,'Press q to capture this emotion', (fX, fY - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        
    else: continue

    cv2.imshow('Emotion_classifier', frameClone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# # import required modules
# from pydub import AudioSegment
# from pydub.playback import play
  
# # for playing wav file
# song = AudioSegment.from_wav("beep.wav")
# print('playing sound using  pydub')
# play(song)
from time import sleep

print('Provide Voice Signal After  a Beep sound')
sleep(2)
playsound('beep-06.wav')

# print('speak')

import speech_recognition as sr

r = sr.Recognizer()

speech = sr.Microphone(device_index=0)

with speech as source:
    print("say something!...")
    audio = r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

try:
    recog = r.recognize_google(audio, language = 'en-US')
    print("You said: " + recog)
    message = [recog]

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))




import pandas as pd
import numpy as np

# text preprocessing
from nltk.tokenize import word_tokenize
import re

# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# preparing input to our model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# keras layers
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Number of labels: joy, anger, fear, sadness, neutral
num_classes = 5

# Number of dimensions for word embedding
embed_num_dims = 300

# Max input length (max number of words) 
max_seq_len = 500

class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
#training code starts
# data_train = pd.read_csv('data/data_train.csv', encoding='utf-8')
# data_test = pd.read_csv('data/data_test.csv', encoding='utf-8')

# X_train = data_train.Text
# X_test = data_test.Text

# y_train = data_train.Emotion
# y_test = data_test.Emotion

# data = data_train.append(data_test, ignore_index=True)
# print(data.Emotion.value_counts())
# data.head(6)

# def clean_text(data):
    
#     # remove hashtags and @usernames
#     data = re.sub(r"(#[\d\w\.]+)", '', data)
#     data = re.sub(r"(@[\d\w\.]+)", '', data)
    
#     # tokenization using nltk
#     data = word_tokenize(data)
    
#     return data


# texts = [' '.join(clean_text(text)) for text in data.Text]

# texts_train = [' '.join(clean_text(text)) for text in X_train]
# texts_test = [' '.join(clean_text(text)) for text in X_test]



# print(texts_train[92])

tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts)

# sequence_train = tokenizer.texts_to_sequences(texts_train)
# sequence_test = tokenizer.texts_to_sequences(texts_test)

# index_of_words = tokenizer.word_index

# # vocab size is number of unique words + reserved 0 index for padding
# vocab_size = len(index_of_words) + 1

# print('Number of unique words: {}'.format(len(index_of_words)))

# X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
# X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )



# encoding = {
#     'joy': 0,
#     'fear': 1,
#     'anger': 2,
#     'sadness': 3,
#     'neutral': 4
# }

# # Integer labels
# y_train = [encoding[x] for x in data_train.Emotion]
# y_test = [encoding[x] for x in data_test.Emotion]


# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# def create_embedding_matrix(filepath, word_index, embedding_dim):
#     vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
#     embedding_matrix = np.zeros((vocab_size, embedding_dim))
#     with open(filepath) as f:
#         for line in f:
#             word, *vector = line.split()
#             if word in word_index:
#                 idx = word_index[word] 
#                 embedding_matrix[idx] = np.array(
#                     vector, dtype=np.float32)[:embedding_dim]
#     return embedding_matrix

# import urllib.request
# import zipfile
# import os

# fname = 'embeddings/wiki-news-300d-1M.vec'

# if not os.path.isfile(fname):
#     print('Downloading word vectors...')
#     urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
#                               'wiki-news-300d-1M.vec.zip')
#     print('Unzipping...')
#     with zipfile.ZipFile('wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
#         zip_ref.extractall('embeddings')
#     print('done.')
    
#     os.remove('wiki-news-300d-1M.vec.zip')



# embedd_matrix = create_embedding_matrix(fname, index_of_words, embed_num_dims)

# new_words = 0

# for word in index_of_words:
#     entry = embedd_matrix[index_of_words[word]]
#     if all(v == 0 for v in entry):
#         new_words = new_words + 1

# print('Words found in wiki vocab: ' + str(len(index_of_words) - new_words))
# print('New words found: ' + str(new_words))

# embedd_layer = Embedding(vocab_size,
#                          embed_num_dims,
#                          input_length = max_seq_len,
#                          weights = [embedd_matrix],
#                          trainable=False)

# kernel_size = 3
# filters = 256

# model = Sequential()
# model.add(embedd_layer)
# model.add(Conv1D(filters, kernel_size, activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))



# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# model.summary()

# batch_size = 256
# epochs = 6

# hist = model.fit(X_train_pad, y_train, 
#                  batch_size=batch_size,
#                  epochs=epochs,
#                  validation_data=(X_test_pad,y_test))


# # Accuracy plot
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# # Loss plot
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()



# predictions = model.predict(X_test_pad)
# predictions = np.argmax(predictions, axis=1)
# predictions = [class_names[pred] for pred in predictions]

# print("Accuracy: {:.2f}%".format(accuracy_score(data_test.Emotion, predictions) * 100))
# print("\nF1 Score: {:.2f}".format(f1_score(data_test.Emotion, predictions, average='micro') * 100))

# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     '''
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     '''
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'

#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     fig, ax = plt.subplots()
    
#     # Set size
#     fig.set_size_inches(12.5, 7.5)
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     ax.grid(False)
    
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax




# print("\nF1 Score: {:.2f}".format(f1_score(data_test.Emotion, predictions, average='micro') * 100))

# # Plot normalized confusion matrix
# plot_confusion_matrix(data_test.Emotion, predictions, classes=class_names, normalize=True, title='Normalized confusion matrix')
# plt.show()

# print('Message: {}\nPredicted: {}'.format(X_test[4], predictions[4]))
#---------------------------------------------------------------

import time
import os

from keras.models import load_model
model = load_model('models/cnn_w2v (copy).h5')

# message = ["I'm feeling cool"]

seq = tokenizer.texts_to_sequences(message)
padded = pad_sequences(seq, maxlen=max_seq_len)

start_time = time.time()
pred = model.predict(padded)

print('Message: ' + str(message))
print('predicted: {} ({:.2f} seconds)'.format(class_names[np.argmax(pred)], (time.time() - start_time)))
predicted = class_names[np.argmax(pred)]
Voice_emotion = ''
# model.save('models/cnn_w2v.h5')
# class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']

if predicted == 'joy':
    Voice_emotion = 'happy'
if predicted == 'fear':
    Voice_emotion = 'sad'
if predicted == 'anger':
    Voice_emotion = 'sad'
if predicted == 'sadness':
    Voice_emotion = 'sad'
if predicted == 'neutral':
    Voice_emotion = 'happy'


import random

print(Face_emotion,Voice_emotion)

if Face_emotion == Voice_emotion:
    #play song from folder
    if Face_emotion == 'happy':
        list_of_happy_songs = os.listdir('happy')
        song = random.choices(list_of_happy_songs)
        # print(song)
        # print('happy')
        playsound('happy/'+song[0])
    else:
        list_of_sad_songs = os.listdir('sad')
        song = random.choices(list_of_sad_songs)
        # print(song)
        # print('sad')

        playsound('sad/'+song[0])

    print('All OK')
else:

    print("Face Emotion and Voice Emotion Does not Match")



