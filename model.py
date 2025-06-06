import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier, accuracy
import pickle

data = pd.read_csv('./combined_emotion.csv')
print('Data loaded successfully!')

joy_data = data[data['emotion'] == 'joy'].head(5000)
sad_data = data[data['emotion'] == 'sad'].head(5000)
anger_data = data[data['emotion'] == 'anger'].head(5000)
fear_data = data[data['emotion'] == 'fear'].head(5000)
love_data = data[data['emotion'] == 'love'].head(5000)
suprise_data = data[data['emotion'] == 'suprise'].head(5000)
data = pd.concat([joy_data, sad_data, anger_data, fear_data, love_data, suprise_data])
print('Data filtered successfully!')

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
  stemmer = SnowballStemmer('english')
  lemmatizer = WordNetLemmatizer()

  tokens = word_tokenize(text)

  tokens = [word.lower() for word in tokens]
  tokens = [word for word in tokens if word not in string.punctuation]
  tokens = [word for word in tokens if word not in stopwords.words('english')]
  tokens = [stemmer.stem(word) for word in tokens]
  tag = pos_tag(tokens)
  tokens = [lemmatizer.lemmatize(word, get_pos(t)) for (word,t) in tag]

  return tokens

data = data.sample(frac=1).reset_index(drop=True)

print('Preprocessing data...')
x = data['sentence']
y = data['emotion']
all_sentences = ' '.join(x)
all_tokens = preprocess(all_sentences)
print('Data preprocessed successfully!')

print('Extracting features...')
fd = FreqDist(all_tokens)

def extract_features(text):
  features = {}

  for word in fd.keys():
    features[word] = (word in text)

  return features

feature_set = [(extract_features(preprocess(sentence)), emotion) for (sentence, emotion) in zip(x,y)]
print('Extracting features Completed')

print('Training...')
training_count = int(len(data['sentence']) * 0.8)
training_set = feature_set[:training_count]
testing_set = feature_set[training_count:]

classifier = NaiveBayesClassifier.train(training_set)
acc = accuracy(classifier, testing_set)
print('Training complete!')

print(f'Accuracy = {acc * 100:.2f}')

print('model dump')
with open('naivebayes_emotion_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
print('Model dumped successfully!')