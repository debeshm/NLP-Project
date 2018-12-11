
# coding: utf-8

# In[1]:
import _pickle as cPickle
from keras.models import Model
import keras
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
import random


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
#from wordcloud import WordCloud
import gensim
from gensim.models import word2vec
import logging
from sklearn.cluster import KMeans

import  tensorflow as tf
from collections import Counter
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import urllib.request
import os
import sys
import zipfile
import logging
import pydot
import graphviz
import re
import json
import ast


# In[2]:


# **********************************************************************
# Reading a pre-trained word embedding and addapting to our vocabulary:
# **********************************************************************





def load_glove():
    embeddings_index = {}
    #f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    #f = open('glove.6B.100d.txt', encoding = 'utf8')
    f = open('gradsq.txt', encoding = 'utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs 
    f.close()
    return embeddings_index

def init_stopwords():
    
    lem = WordNetLemmatizer()

#https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
    APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
    }

    eng_stopwords = set(stopwords.words("english"))
    
    user_stop_words = []
    #print("Eng stopwords before user stopwords:: ", len(eng_stopwords))
    
    for w in user_stop_words:
        if w not in eng_stopwords:
            eng_stopwords.add(w)
    
    #print("Eng stopwords after user stopwords:: ",len(eng_stopwords))

    return lem, APPO, eng_stopwords

def pre_process(text, lem, APPO, eng_stopwords):
        
    text=text.lower()   
    
    text = re.sub("was good.", " ", text)
    text = re.sub("was bad.", " ", text)
    
    words = word_tokenize(text)
    
    # (')aphostophe  replacement (ie)   you're --> you are
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    
    # remove punctuations 
    words = [w.lower() for w in words if w.isalpha()]
    
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent = " ".join(words)
    return clean_sent

# **********************************************************************
# Developing our vocabulary from the dataset:
# **********************************************************************

def load_vocab(self):
    
    inp_sentences = self.input_texts
    out_sentences = self.target_texts
    
    print("input sentences :: \n ",inp_sentences[0:5])
    print("output sentences :: \n ",out_sentences[0:5])
    print(type(inp_sentences))

    inp_model = word2vec.Word2Vec(inp_sentences, iter=5, min_count=5, size=50, workers=4)
    out_model = word2vec.Word2Vec(out_sentences, iter=5, min_count=5, size=50, workers=4)

    ########   Input embedding matrix ##########
    
    inp_vocab_size = len(inp_model.wv.vocab)
    print(inp_vocab_size)
    # get the most common words
    print("Most common words :: ", inp_model.wv.index2word[0:10])
    # get the least common words
    print("Least common words :: ", inp_model.wv.index2word[inp_vocab_size - 10:inp_vocab_size-1])
    


    f=open('vectors.txt')
    vector_embd={}
    for line in f:
        values=line.split()
        word=values[0]
        coefs=np.asarray(values[1:], dtype='float32')
        vector_embd[word]=coefs


    # convert the wv word vectors into a numpy matrix 
    inp_embedding_matrix = {} 
    #inp_embedding_matrix=vector_embd


    for i, word in enumerate(inp_model.wv.vocab):
        inp_embedding_vector = inp_model.wv[inp_model.wv.index2word[i]]
        if inp_embedding_vector is not None:
            inp_embedding_matrix[word] = inp_embedding_vector
    
            
    ########   Output embedding matrix ##########
   
    out_vocab_size = len(out_model.wv.vocab)
    print(out_vocab_size)
    # get the most common words
    print(out_model.wv.index2word[0:10])
    # get the least common words
    print(out_model.wv.index2word[out_vocab_size - 10:out_vocab_size-1])
    
    # convert the wv word vectors into a numpy matrix 
    out_embedding_matrix = {} 

    #out_embedding_matrix=vector_embd
    

    for i, word in enumerate(out_model.wv.vocab):
        out_embedding_vector = out_model.wv[out_model.wv.index2word[i]]
        if out_embedding_vector is not None:
            out_embedding_matrix[word] = out_embedding_vector

    
    embeddings_index = {}
    #f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    #f = open('glove.6B.100d.txt', encoding = 'utf8')
    f = open('vectors.txt')
    for line in f:
        values = line.split()
        word = values[0]
        if word in ('start','end'):
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs 
        
    f.close()
    #out_embedding_matrix['start']=embeddings_index['start']
    #out_embedding_matrix['end']=embeddings_index['end']

    return inp_embedding_matrix, out_embedding_matrix

# **********************************************************************
# Reading input text and the replies
# **********************************************************************
def read_input():
    '''
    df = pd.read_csv(DATA, encoding = 'latin-1')
    
    input1 = df['SentimentText'].fillna("")
    output1 = df['ResponseText'].fillna("")
    '''

    lem, APPO, eng_stopwords = init_stopwords()
    #clean_input_text = pre_process(str(input_text), lem, APPO, eng_stopwords)


    with open('email_summarized_response.json','r') as f:
        inp = json.load(f)

    with open("questions.txt") as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    questions = [x.strip() for x in content] 
    #print(questions)

    with open("reply.txt") as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    reply = [x.strip() for x in content] 

    inp = ast.literal_eval(json.dumps(inp))
    df = {'SentimentText' : [], 'ResponseText': []}

    '''
    for i in range(len(questions)):
        df['SentimentText'].append(pre_process(str(questions[i]), lem, APPO, eng_stopwords))
        df['ResponseText'].append(pre_process(str(reply[i]), lem, APPO, eng_stopwords))
    '''

    for key in inp:
        if len(inp[key])>1:
            df['SentimentText'].append(pre_process(str(inp[key][0]), lem, APPO, eng_stopwords))
            df['ResponseText'].append(pre_process(str(inp[key][1]), lem, APPO, eng_stopwords))
    '''
            df['SentimentText'].append(inp[key][0])
            df['ResponseText'].append(inp[key][1])
    '''
    
    df['SentimentText']=df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']+df['SentimentText']
    df['ResponseText']=df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']+df['ResponseText']
    '''
    for key in inp:
        print(key)
        print(inp[key])
        for r in inp[key]:
            df['SentimentText'].append(key)
            df['ResponseText'].append(r)

    '''

    #print("INPUT:"+str(len(df['SentimentText'])))
    #print("OUTPUT"+str(len(df['ResponseText'])))

    df['SentimentText'] = pd.Series(df['SentimentText']).astype(str)
    df['ResponseText'] = pd.Series(df['ResponseText']).astype(str)
    input1 = df['SentimentText'].fillna("")
    output1 = df['ResponseText'].fillna("")
    
    print(len(df['SentimentText']))
    
    
    # pre-processing only the input data 
    lem, APPO, eng_stopwords = init_stopwords()
    inp_clean_corpus = input1.apply(lambda x: pre_process(str(x), lem, APPO, eng_stopwords))
    
    print(type(inp_clean_corpus))
    
#    vocab = load_vocab(inp_clean_corpus)
          
    # some stats about input data
    num_words = 10000
    tokenize = Tokenizer(num_words=num_words)
    tokenize.fit_on_texts(inp_clean_corpus)
    
    tok_inp = tokenize.texts_to_sequences(inp_clean_corpus)
    
    inp_len = [len(words) for words in tok_inp]
    mod_tok_inp = int(np.mean(inp_len) + 2 * np.std(inp_len))
    print("\nmean:: ", np.mean(inp_len))
    print("max:: ", np.max(inp_len))
    print("inp token size - average + 2*sd --> :: ", (mod_tok_inp))
    print("inp - total inp tokens:: ", len(inp_len))
    
    final_len_inp = [(tok) for tok in inp_len if tok < mod_tok_inp]
    final_len_inp = list(final_len_inp)
    print("inp - no of tokens < mod_tok :: ", len(final_len_inp))
    
    
    input1 = inp_clean_corpus.tolist()
    output1 = output1.tolist()
    
    target_counter = Counter()
    
    input_texts = []
    target_texts = []
    
    print(type(input1))
    print("input:: \n", input1[1:5])
    
    print(type(output1))
    print("output:: \n", output1[1:5])

    for line in input1:
      
        inp_words = [w.lower() for w in nltk.word_tokenize(line)]
        if len(inp_words) > MAX_TARGET_SEQ_LENGTH:
            inp_words = inp_words[0:MAX_TARGET_SEQ_LENGTH]
    
        input_texts.append(inp_words)

    for line1 in output1:
        out_words = [w.lower() for w in nltk.word_tokenize(line1)]
        if len(out_words) > MAX_TARGET_SEQ_LENGTH:
            out_words = out_words[0:MAX_TARGET_SEQ_LENGTH]
    
        tar_words = out_words[:]
        tar_words.insert(0, 'start')
        tar_words.append('end')
        for w in tar_words:
            target_counter[w] += 1
        target_texts.append(tar_words)
    
    
    print("\n Input texts :: \n\n ", input_texts[1:5])
    print("\n Target texts :: \n\n ",target_texts[1:5])

    return input_texts, target_texts, target_counter

def get_target(self):
   
    target_word2idx = dict()   
    for idx, word in enumerate(self.target_counter.most_common(MAX_VOCAB_SIZE)):
        target_word2idx[word[0]] = idx + 1
    
    if 'UNK' not in target_word2idx:
        target_word2idx['UNK'] = 0
    
    target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

    num_decoder_tokens = len(target_idx2word)+1
    
    input_texts_word2em = []
    
    encoder_max_seq_length = 0
    decoder_max_seq_length = 0
    
    for input_words, target_words in zip(self.input_texts, self.target_texts):
        encoder_input_wids = []
        for w in input_words:
            emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
            if w in self.word2em_input:
                emb = self.word2em_input[w]
            encoder_input_wids.append(emb)
    
        input_texts_word2em.append(encoder_input_wids)
        encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
        decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)
    
    #print("input_texts_word2em for first 2 sentenses:: \n", input_texts_word2em[1:3])
    
    context = dict()
    context['num_decoder_tokens'] = num_decoder_tokens
    context['encoder_max_seq_length'] = encoder_max_seq_length
    context['decoder_max_seq_length'] = decoder_max_seq_length
    
    return target_word2idx, target_idx2word, context, input_texts_word2em
    
def generate_batch(input_word2em_data, output_text_data, self):
    
    num_batches = len(input_word2em_data) // BATCH_SIZE
    print("context:: \n", self.context)
    print("len of input data :: ", len(input_word2em_data))
    print("num of batches :: ", num_batches)
    
    while True:
        for batchIdx in range(0, num_batches):
            
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_word2em_data[start:end], maxlen=self.context['encoder_max_seq_length'])
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, self.context['decoder_max_seq_length'], self.num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, self.context['decoder_max_seq_length'], GLOVE_EMBEDDING_SIZE))
            
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = self.target_word2idx['UNK']  # default UNK
                    if w in self.target_word2idx:
                        w2idx = self.target_word2idx[w]
                    if w in self.word2em_output:
                        '''
                        if w in ['start', 'end']:
                    
                            print(w)
                            print(decoder_input_data_batch.shape)
                            print(self.word2em_output[w].shape)
                            
                            continue
                            '''
                        decoder_input_data_batch[lineIdx, idx, :] = self.word2em_output[w]
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch



# In[ ]:




# In[ ]:


class Cluster:
    def __init__(self, X):
        self.kmeans=KMeans(n_clusters=5, random_state=0).fit(X)

    def get_prediction(self, x1, x2):
        self.kmeans.predict(x1, x2)


class CornellWordGloveChatBot(object):
    model = None
    encoder_model = None
    decoder_model = None
    target_counter = None
    target_word2idx = None
    target_idx2word = None
    max_decoder_seq_length = None
    max_encoder_seq_length = None
    num_decoder_tokens = None
    word2em_input = None
    word2em_output = None
    accr=0
    
    context = None
    input_texts = None
    target_texts = None
    
    def __init__(self):
        config = tf.ConfigProto( device_count = {'GPU': 4 , 'CPU': 8} )
        sess = tf.Session(config=config) 
        keras.backend.set_session(sess)

        self.word2em = load_glove()
        
        self.input_texts, self.target_texts, self.target_counter = read_input()
        print("input texts:: \n", self.input_texts[0:3])
        
        self.word2em_input , self.word2em_output = load_vocab(self)


        self.create_reply_clusters()
        
        print("Length of input word2em :: ", len(self.word2em_input))
        print("Length of output word2em :: ", len(self.word2em_output))
        
        for idx, (input_words, target_words) in enumerate(zip(self.input_texts, self.target_texts)):
            if idx > 10:
                break
                print([input_words, target_words])

        self.target_word2idx, self.target_idx2word , self.context, input_texts_word2em = get_target(self)
        
        self.max_encoder_seq_length = self.context['encoder_max_seq_length']
        self.max_decoder_seq_length = self.context['decoder_max_seq_length']
        self.num_decoder_tokens = self.context['num_decoder_tokens']
        
        print("context: ",self.context)
        
        encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        encoder_lstm1 = LSTM(units=HIDDEN_UNITS, return_state=True, name="encoder_lstm1" , dropout=0.2)
        #logger.info("Added LSTM Layer")

        #encoder_lstm2 = LSTM(units=HIDDEN_UNITS, return_state=True,  name="encoder_lstm2", dropout=0.2)
        #encoder_lstm3 = LSTM(units=HIDDEN_UNITS, return_state=True, name="encoder_lstm3")
        
        #x = encoder_lstm1(encoder_inputs)
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm1(encoder_inputs)
        
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm', dropout=0.2)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        #self.model = keras.utils.multi_gpu_model(self.model, 2)#GPU DEBESH
        gpu_list = ['/xla_gpu:0', '/gpu:0']

        print(self.model.summary())
        
        #plot_model(self.model, to_file='RNN_model.png', show_shapes=True)
        
        self.model.load_weights('word-glove-weights_epoch300_2.h5')

        
        
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(input_texts_word2em, self.target_texts, test_size=0.2, random_state=42)

        print("Length of train data:: ", len(Xtrain))
        print("Length of test data:: ", len(Xtest))
        
        train_gen = generate_batch(Xtrain, Ytrain, self)
        test_gen = generate_batch(Xtest, Ytest, self)
        
        train_num_batches = len(Xtrain) // BATCH_SIZE
        test_num_batches = len(Xtest) // BATCH_SIZE
        print("TEST NUM: "+str(len(Xtest)))
        print("BATCH Size: "+str(BATCH_SIZE))
        print("TEST NUM BATCHES: "+str(test_num_batches))
        
        #checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
        '''
        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches ) #, callbacks=[checkpoint])        
        
        self.model.save_weights(WEIGHT_FILE_PATH)
        '''
        
        
        self.encoder_model = Model(encoder_inputs, encoder_states)
        #(self.encoder_model, to_file='RNN_encoder_model.png', show_shapes=True)
        
        
        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
        #plot_model(self.decoder_model, to_file='RNN_decoder_model.png', show_shapes=True)



    def create_reply_clusters(self):
        X, seq_len=(self.get_reply_embeddings())
        X = np.asarray(X)
        # X.reshape(544, seq_len)
        #print(X.shape)
        kmeans=KMeans(n_clusters=5, random_state=0).fit(X)
        with open('classifier.pkl', 'wb') as fid:
            cPickle.dump(kmeans, fid)


    def get_reply_embeddings(self):
        self.reply_cluster_length = 0
        target_texts_word2em = []
        decoder_max_seq_length=0
        for input_words, target_words in zip(self.input_texts, self.target_texts):
            encoder_target_wids = []
            for w in target_words:
                emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
                if w in self.word2em_output:
                    emb = self.word2em_output[w]
                encoder_target_wids.append(emb)
            self.reply_cluster_length=max(len(target_words), self.reply_cluster_length)
            target_texts_word2em.append(encoder_target_wids)
        for i in range(len(target_texts_word2em)):
            while len(target_texts_word2em[i]) < self.reply_cluster_length:
                emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
                target_texts_word2em[i].append(emb)
            target_texts_word2em[i]=np.asarray(target_texts_word2em[i])
            target_texts_word2em[i]=target_texts_word2em[i].flatten()
            # if len(target_texts_word2em[i]) < decoder_max_seq_length:
            #     target_texts_word2em[i] = pad_sequences(target_texts_word2em[i][:],maxlen=decoder_max_seq_length)
            #print("padding",target_texts_word2em[i])
        return target_texts_word2em, self.reply_cluster_length

        
    def reply(self, input_text, input_reply):
        input_seq = []
        input_emb = []
        print("input text:: \n\n ", input_text)
        
        # pre-processing only the input data 
        lem, APPO, eng_stopwords = init_stopwords()
        clean_input_text = pre_process(str(input_text), lem, APPO, eng_stopwords)
        #print("CLEAN INPUT:")
        #print(clean_input_text)
        
        for word in nltk.word_tokenize(clean_input_text.lower()):
            emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
            if word in self.word2em_input:
                emb = self.word2em_input[word]
            input_emb.append(emb)
        input_seq.append(input_emb)
    
        #print("INPUT SEQ")
        #print(len(input_seq[0]))
        
        input_seq = pad_sequences(input_seq, dtype='float32', maxlen=self.max_encoder_seq_length)

        #print("INPUT SEQ")
        #print(input_seq.shape)
               
        states_value = self.encoder_model.predict(input_seq)

        #print("STATES VALUE")
        #print(states_value)

        target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
        target_seq[0, 0, :] = self.word2em_output['start'] #I COMMENTED THIS, DEBESH

        #print("TARGET SEQUENCE")
        #print(target_seq)
        
        target_text = ''
        target_text_len = 0
        terminated = False

        predicted_target_embedding=[]

        while not terminated:
            
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            #print("output tokens shape  :: \n\n ", output_tokens.shape)
            
            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            
            sample_word = self.target_idx2word[sample_token_idx]

            predicted_target_embedding.append(self.word2em_output[sample_word])


            target_text_len += 1
        
            if sample_word != 'start' and sample_word != 'end':
                #print("sample word :: ", sample_word)
                target_text += ' ' + sample_word
            
            if sample_word == 'end' or target_text_len >= self.max_decoder_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
            if sample_word in self.word2em_output:
                target_seq[0, 0, :] = self.word2em_output[sample_word]

            states_value = [h, c]

        lem, APPO, eng_stopwords = init_stopwords()
        input_reply =pre_process(input_reply, lem, APPO, eng_stopwords)


        actual_target_embedding = []
        for w in input_reply:
            emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
            if w in self.word2em_output:
                emb = self.word2em_output[w]
            actual_target_embedding.append(emb)
        while len(actual_target_embedding) < self.reply_cluster_length:
            emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
            actual_target_embedding.append(emb)

        while len(predicted_target_embedding) < self.reply_cluster_length:
            emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
            predicted_target_embedding.append(emb)

        actual_target_embedding=np.asarray(actual_target_embedding[:self.reply_cluster_length]).flatten()
        predicted_target_embedding=np.asarray(predicted_target_embedding).flatten()

        km=None
        with open('classifier.pkl', 'rb') as f:
            km=cPickle.load(f)

        cluster_pred=km.predict([predicted_target_embedding.tolist(),actual_target_embedding.tolist()])
        #print("Cluster prediction",km.predict([predicted_target_embedding.tolist(),actual_target_embedding.tolist()]))
        if cluster_pred[0] == cluster_pred[1]:
            self.accr+=1
        #print(cluster_pred[0] == cluster_pred[1])
        #print("ACCR", self.accr)
        return ''.join(list(OrderedDict.fromkeys(target_text.strip())))
        #return target_text.strip()

    def test_run(self):
        '''
        print(self.reply('Not so good experience. Washroom was not cleaned properly and room service was not quick to resond.'))
        print(self.reply('Hotel was ok. Food was good and staff was very cooperative in providing services.'))
        print(self.reply('I loved the environment of the hotel !!!. It was great living there '))
        '''

        c=0
        print(self.reply("Bob The application for the variance states that I applied for a permit on October 29th and was rejected Do the discussions with the building inspector constitute denial Phillip Original Message", "Your ideas about opening up the kitchento the living room or dining room would still be possible You could convert the modified bedroom into a media room and usethe new space for your game play music room"))
        print(self.reply("Really look forward to seeing you Bob What time", "or any of its affiliates and the intended recipient or any otherparty and may not be relied on by anyone as the basis of a contract byestoppel or otherwise This e mail and any attachmentshereto are not intended to be an offer or an acceptance and do notcreate or evidence a binding and enforceable contract between EnronCorp"))
        print(self.reply("Call me after you get this message Jeff SmithThe Smith Company2714 Bee Cave Road Suite 100 DAustin Texas Forwarded by Phillip K Allen HOU ECT on 02 09 2001I am sending the Dr a contract for Monday delivery", "Please confirm receipt I just refaxed"))


        with open('email_response.json','r') as f:
            inp = json.load(f)


        for key in inp:
            if len(inp[key])>1:
                if inp[key] is not '':
                    if bool(inp[key][0]):
                        c+=1
                        print(self.reply(inp[key][0], inp[key][1]))
                '''
                df['SentimentText'].append(inp[key][0])
                df['ResponseText'].append(inp[key][1])
                '''
        
        c+=3

        print("ACCURACY: "+str((self.accr/float(c))*100)+"%")


        while True:
            print(self.reply(input("Enter email:"), ''))
        

def main():
    np.random.seed(42)

    model = CornellWordGloveChatBot()
    model.test_run()
    
if __name__ == '__main__':
    
    MAX_VOCAB_SIZE = 10000
    BATCH_SIZE = 64
    NUM_EPOCHS = 1000
    #GLOVE_EMBEDDING_SIZE = 100
    GLOVE_EMBEDDING_SIZE = 50

    HIDDEN_UNITS = 32
    MAX_INPUT_SEQ_LENGTH = 150
    MAX_TARGET_SEQ_LENGTH = 150
    
    DATA_SET_NAME = 'cornell'
    DATA = 'D:/CBA/Sessions/Capstone/Data/PositiveOnly.csv'
    DATA_PATH = 'movie_lines_cleaned_10k.txt'
    
    WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'
    #WEIGHT_FILE_PATH = 'D:/CBA/word-glove-weights.h5'
    WEIGHT_FILE_PATH = 'word-glove-weights_epoch300_2.h5'
    
    main()
    
    


# In[ ]: