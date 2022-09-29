from pipeline import eval 
from pipeline import LogReg, NaiveBayes, RandFrst, SVM, DescTree
from pipeline2 import LogReg2, NaiveBayes2, RandFrst2, SVM2, DescTree2
from data_clean import process_df, clean, tknzr, stmr, lmtzr 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


import nltk
from nltk.tokenize import RegexpTokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.model_selection import GridSearchCV, train_test_split

from collections import Counter

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

data = pd.read_csv('combined_data.csv')

labels = {'negative':0, 'neutral':1, 'positive':2}
data['label'] = [labels[item] for item in data.sntmt]

opt = input('Choose a feature extraction technique: '
                '\n\n\t "CV" for Count Vectorizer'
                '\n\t "TF" for TFIDF Vectorizer'
                '\n\t "W2V" for Word2Vec method'
                '\n\t "D2V" for Doc2Vec method \n\n' )

###WITH STOP WORDS###
#no pre-processing
data_sw_og = process_df(data.copy(), tknzr, rem_sw=False)
#stemming 
data_sw_st = process_df(data.copy(), stmr, rem_sw=False)
#lemmatization
data_sw_lm = process_df(data.copy(), lmtzr, rem_sw=False)

###WITHOUT STOP WORDS###
#remove stop words only
data_nsw_og = process_df(data.copy(), tknzr, rem_sw=True)
#remove stop words + stemming 
data_nsw_st = process_df(data.copy(), stmr, rem_sw=True)
#remove stop words + lemmatization
data_nsw_lm = process_df(data.copy(), lmtzr, rem_sw=True)

DATASETS = [data_sw_og, data_sw_st, data_sw_lm, data_nsw_og, data_nsw_st, data_nsw_lm]

n_grm = [(1,2),(1,3),(2,2),(2,3),(3,3)]
clfrs = ['LogReg','RandFrst','NaiveBayes','SVM','DescTree']
clfrs2 = ['LogReg','RandFrst','SVM','DescTree']
features = [2000, 4000]


RESULTS_COUNTVEC = RESULTS_TFIDF = {'Stop_Words':[],
                                    'Pre-Processing':[],
                                    'N_Grams':[],
                                    'No_of_Features':[],
                                    'Classifier':[],
                                    'Accuracy':[],
                                    'Precision':[],
                                    'Recall':[],
                                    'F-Score':[]
                                    }

RESULTS_WRD2VEC = RESULTS_DOC2VEC = {'Stop_Words':[],
                                     'Pre-Processing':[],
                                     'Model': [],
                                     'No_of_Features':[],
                                     'Classifier':[],
                                     'Accuracy':[],
                                     'Precision':[],
                                     'Recall':[],
                                     'F-Score':[]
                                    }

if opt == 'CV':

    for ds in DATASETS:

        for ng in n_grm:

            count_vec = CountVectorizer(ngram_range=ng)
            count_data = count_vec.fit_transform(ds.processed_text)
            count_vec_df = pd.DataFrame(count_data.toarray(), columns = count_vec.get_feature_names_out())

            x = count_vec_df.values
            y = ds['label'].values

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=15)

            for clfr in clfrs:

                if clfr == 'LogReg':
                    clf = LogReg
                elif clfr == 'RandFrst':
                    clf = RandFrst
                elif clfr == 'NaiveBayes':
                    clf = NaiveBayes
                elif clfr == 'SVM':
                    clf = SVM
                elif clfr == 'DescTree':
                    clf = DescTree
                    
                for nf in features:

                    print("########## RESULTS FOR {} MODEL, {} NGRAMS, {} FEATURES  ##########".format(clfr, ng, nf))
                    y_pred = clf(X_train, X_test, y_train, y_test, nf)
                    a,p,r,f = eval(y_test, y_pred)

                    if ds.equals(data_sw_og):
                        RESULTS_COUNTVEC['Stop_Words'].append('Not Removed')
                        RESULTS_COUNTVEC['Pre-Processing'].append('None')
                    elif ds.equals(data_sw_st):
                        RESULTS_COUNTVEC['Stop_Words'].append('Not Removed')
                        RESULTS_COUNTVEC['Pre-Processing'].append('Stemming')
                    elif ds.equals(data_sw_lm):
                        RESULTS_COUNTVEC['Stop_Words'].append('Not Removed')
                        RESULTS_COUNTVEC['Pre-Processing'].append('Lemmatizing')
                    elif ds.equals(data_nsw_og):
                        RESULTS_COUNTVEC['Stop_Words'].append('Removed')
                        RESULTS_COUNTVEC['Pre-Processing'].append('None')
                    elif ds.equals(data_nsw_st):
                        RESULTS_COUNTVEC['Stop_Words'].append('Removed')
                        RESULTS_COUNTVEC['Pre-Processing'].append('Stemming')
                    elif ds.equals(data_nsw_lm):
                        RESULTS_COUNTVEC['Stop_Words'].append('Removed')
                        RESULTS_COUNTVEC['Pre-Processing'].append('Lemmatizing')
                    RESULTS_COUNTVEC['N_Grams'].append(ng)
                    RESULTS_COUNTVEC['No_of_Features'].append(nf)
                    RESULTS_COUNTVEC['Classifier'].append(clfr)
                    RESULTS_COUNTVEC['Accuracy'].append(a)
                    RESULTS_COUNTVEC['Precision'].append(p)
                    RESULTS_COUNTVEC['Recall'].append(r)
                    RESULTS_COUNTVEC['F-Score'].append(f)


        results_cntvec = pd.DataFrame(RESULTS_COUNTVEC)
        sorted_results_cntvec = results_cntvec.sort_values(by=['F-Score'], ascending = False)
        sorted_results_cntvec.to_csv('./Results/CountVec_Results.csv', index=False)

if opt == 'TF':

    for ds in DATASETS:

        for ng in n_grm:

            tfidf_vec = TfidfVectorizer(ngram_range=ng)
            tfidf_data = tfidf_vec.fit_transform(ds.processed_text)
            tfidf_vec_df = pd.DataFrame(tfidf_data.toarray(), columns = tfidf_vec.get_feature_names_out())

            x = tfidf_vec_df.values
            y = ds['label'].values

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=15)

            for clfr in clfrs:

                if clfr == 'LogReg':
                    clf = LogReg
                elif clfr == 'RandFrst':
                    clf = RandFrst
                elif clfr == 'NaiveBayes':
                    clf = NaiveBayes
                elif clfr == 'SVM':
                    clf = SVM
                elif clfr == 'DescTree':
                    clf = DescTree
                    
                for nf in features:

                    print("########## RESULTS FOR {} MODEL, {} NGRAMS, {} FEATURES  ##########".format(clfr, ng, nf))
                    y_pred = clf(X_train, X_test, y_train, y_test, nf)
                    a,p,r,f = eval(y_test, y_pred)

                    if ds.equals(data_sw_og):
                        RESULTS_TFIDF['Stop_Words'].append('Not Removed')
                        RESULTS_TFIDF['Pre-Processing'].append('None')
                    elif ds.equals(data_sw_st):
                        RESULTS_TFIDF['Stop_Words'].append('Not Removed')
                        RESULTS_TFIDF['Pre-Processing'].append('Stemming')
                    elif ds.equals(data_sw_lm):
                        RESULTS_TFIDF['Stop_Words'].append('Not Removed')
                        RESULTS_TFIDF['Pre-Processing'].append('Lemmatizing')
                    elif ds.equals(data_nsw_og):
                        RESULTS_TFIDF['Stop_Words'].append('Removed')
                        RESULTS_TFIDF['Pre-Processing'].append('None')
                    elif ds.equals(data_nsw_st):
                        RESULTS_TFIDF['Stop_Words'].append('Removed')
                        RESULTS_TFIDF['Pre-Processing'].append('Stemming')
                    elif ds.equals(data_nsw_lm):
                        RESULTS_TFIDF['Stop_Words'].append('Removed')
                        RESULTS_TFIDF['Pre-Processing'].append('Lemmatizing')
                    RESULTS_TFIDF['N_Grams'].append(ng)
                    RESULTS_TFIDF['No_of_Features'].append(nf)
                    RESULTS_TFIDF['Classifier'].append(clfr)
                    RESULTS_TFIDF['Accuracy'].append(a)
                    RESULTS_TFIDF['Precision'].append(p)
                    RESULTS_TFIDF['Recall'].append(r)
                    RESULTS_TFIDF['F-Score'].append(f)


        results_tfidf = pd.DataFrame(RESULTS_TFIDF)
        sorted_results_tfidf = results_tfidf.sort_values(by=['F-Score'], ascending = False)
        sorted_results_tfidf.to_csv('./Results/Tfidf_Results.csv', index=False)

if opt == 'W2V':

    features2 = [300,600,900]

    def word_vector(tokens, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in tokens:
            try:
                vec += model_w2v.wv[word].reshape((1, size))
                count += 1.
            except KeyError:  # handling the case where the token is not in vocabulary
                continue
        if count != 0:
            vec /= count
        return vec

    for ds in DATASETS:

        for sg in [0,1]:

            ds['tokens'] = ds['processed_text'].map(lambda txt: tknzr(txt, False).split(" "))

            tokens = pd.Series(ds['tokens']).values

            model_w2v = Word2Vec(
                tokens,
                vector_size=1000, # desired no. of features/independent variables
                window=3, # context window size
                min_count=1, # Ignores all words with total frequency lower than 2.                                  
                sg = sg, # 1 for skip-gram model
                hs = 0,
                negative = 10, # for negative sampling
                workers= 32, # no.of cores
                seed = 34
            )

            wordvec_arrays = np.zeros((len(tokens), 1000)) 
            for i in range(len(tokens)):
                wordvec_arrays[i,:] = word_vector(tokens[i], 1000)
            wordvec_df = pd.DataFrame(wordvec_arrays)

            x = wordvec_df.values
            y = ds['label'].values

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=15)

            for clfr in clfrs2:

                if clfr == 'LogReg':
                    clf = LogReg2
                elif clfr == 'RandFrst':
                    clf = RandFrst2
                elif clfr == 'SVM':
                    clf = SVM2
                elif clfr == 'DescTree':
                    clf = DescTree2

                for nf in features2:
                    print("########## RESULTS FOR {} CLASSIFIER, {} MODEL, {} FEATURES  ##########".format(clfr, sg, nf))
                    print(" 0 => CBOW ; 1 => SkipGram ")
                    y_pred = clf(X_train, X_test, y_train, y_test, nf)
                    a,p,r,f = eval(y_test, y_pred)

                    if ds.equals(data_sw_og):
                        RESULTS_WRD2VEC['Stop_Words'].append('Not Removed')
                        RESULTS_WRD2VEC['Pre-Processing'].append('None')
                    elif ds.equals(data_sw_st):
                        RESULTS_WRD2VEC['Stop_Words'].append('Not Removed')
                        RESULTS_WRD2VEC['Pre-Processing'].append('Stemming')
                    elif ds.equals(data_sw_lm):
                        RESULTS_WRD2VEC['Stop_Words'].append('Not Removed')
                        RESULTS_WRD2VEC['Pre-Processing'].append('Lemmatizing')
                    elif ds.equals(data_nsw_og):
                        RESULTS_WRD2VEC['Stop_Words'].append('Removed')
                        RESULTS_WRD2VEC['Pre-Processing'].append('None')
                    elif ds.equals(data_nsw_st):
                        RESULTS_WRD2VEC['Stop_Words'].append('Removed')
                        RESULTS_WRD2VEC['Pre-Processing'].append('Stemming')
                    elif ds.equals(data_nsw_lm):
                        RESULTS_WRD2VEC['Stop_Words'].append('Removed')
                        RESULTS_WRD2VEC['Pre-Processing'].append('Lemmatizing')
                    
                    if sg == 0:
                        RESULTS_WRD2VEC['Model'].append('CBOW')
                    else:
                        RESULTS_WRD2VEC['Model'].append('SkipGram')

                    RESULTS_WRD2VEC['No_of_Features'].append(nf)
                    RESULTS_WRD2VEC['Classifier'].append(clfr)
                    RESULTS_WRD2VEC['Accuracy'].append(a)
                    RESULTS_WRD2VEC['Precision'].append(p)
                    RESULTS_WRD2VEC['Recall'].append(r)
                    RESULTS_WRD2VEC['F-Score'].append(f)

        results_wrd2vec = pd.DataFrame(RESULTS_WRD2VEC)
        sorted_results_wrd2vec = results_wrd2vec.sort_values(by=['F-Score'], ascending = True)
        sorted_results_wrd2vec.to_csv('./Results/Wrd2Vec_Results1.csv', index=False)

if opt == 'D2V':

    features2 = [300, 600, 900]

    for ds in DATASETS:

        for dm in [0,1]:

            ds['tokens']=ds['processed_text'].map(lambda txt: tknzr(txt, rem_sw=False).split(" "))
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(ds['tokens'])]
            
            d2v_model = Doc2Vec(
                documents=documents,
                vector_size=1000,
                window=3,
                dm=dm,
                min_count=1,
                workers=32,
                negative=10
            )
            
            lines = []
            for idx, row in data_sw_og.iterrows():
                vector = d2v_model.infer_vector(row['tokens'])
                line = [vector_element for vector_element in vector]
                lines.append(line)
            
            doc2vec_df = pd.DataFrame(lines)

            x = doc2vec_df.values
            y = ds['label'].values

            X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, shuffle= True, random_state=15)

            for clfr in clfrs2:

                if clfr == 'LogReg':
                    clf = LogReg2
                elif clfr == 'RandFrst':
                    clf = RandFrst2
                elif clfr == 'SVM':
                    clf = SVM2
                elif clfr == 'DescTree':
                    clf = DescTree2

                for nf in features2:
                    print("########## RESULTS FOR {} CLASSIFIER, {} MODEL, {} FEATURES  ##########".format(clfr, dm, nf))
                    print(" 0 => DBOW ; 1 => Distributed Memory ")
                    y_pred = clf(X_train, X_test, y_train, y_test, nf)
                    a,p,r,f = eval(y_test, y_pred)

                    if ds.equals(data_sw_og):
                        RESULTS_DOC2VEC['Stop_Words'].append('Not Removed')
                        RESULTS_DOC2VEC['Pre-Processing'].append('None')
                    elif ds.equals(data_sw_st):
                        RESULTS_DOC2VEC['Stop_Words'].append('Not Removed')
                        RESULTS_DOC2VEC['Pre-Processing'].append('Stemming')
                    elif ds.equals(data_sw_lm):
                        RESULTS_DOC2VEC['Stop_Words'].append('Not Removed')
                        RESULTS_DOC2VEC['Pre-Processing'].append('Lemmatizing')
                    elif ds.equals(data_nsw_og):
                        RESULTS_DOC2VEC['Stop_Words'].append('Removed')
                        RESULTS_DOC2VEC['Pre-Processing'].append('None')
                    elif ds.equals(data_nsw_st):
                        RESULTS_DOC2VEC['Stop_Words'].append('Removed')
                        RESULTS_DOC2VEC['Pre-Processing'].append('Stemming')
                    elif ds.equals(data_nsw_lm):
                        RESULTS_DOC2VEC['Stop_Words'].append('Removed')
                        RESULTS_DOC2VEC['Pre-Processing'].append('Lemmatizing')
                    
                    if dm == 0:
                        RESULTS_DOC2VEC['Model'].append('DBOW')
                    else:
                        RESULTS_DOC2VEC['Model'].append('Dist. Memory')

                    RESULTS_DOC2VEC['No_of_Features'].append(nf)
                    RESULTS_DOC2VEC['Classifier'].append(clfr)
                    RESULTS_DOC2VEC['Accuracy'].append(a)
                    RESULTS_DOC2VEC['Precision'].append(p)
                    RESULTS_DOC2VEC['Recall'].append(r)
                    RESULTS_DOC2VEC['F-Score'].append(f)

        results_doc2vec = pd.DataFrame(RESULTS_DOC2VEC)
        sorted_results_doc2vec = results_doc2vec.sort_values(by=['F-Score'], ascending = True)
        sorted_results_doc2vec.to_csv('./Results/Doc2Vec_Results.csv', index=False)







            