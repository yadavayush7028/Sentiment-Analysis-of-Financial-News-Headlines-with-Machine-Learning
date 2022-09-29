import numpy as np
import pandas as pd
from pipeline import LogReg, eval
from data_clean import tknzr, stmr, process_df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

trn_df = pd.read_csv('combined_data.csv')
labels = {'negative':0, 'neutral':1, 'positive':2}
trn_df['label'] = [labels[item] for item in trn_df.sntmt]

train_df = process_df(trn_df.copy(), func=stmr, rem_sw=False)

x = train_df['processed_text'].values
y = train_df['label'].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=15)

data=[]
with open('project2_test_data.txt', 'r', encoding='utf8') as f:

    lines = f.readlines()
    for line in lines:
        row = [line]
        data.append(row)

test_data = pd.DataFrame(data, columns=['text'])

tst_df = process_df(test_data, stmr, False)

def LogReg(X_train, X_test, y_train, y_test):
    
  clf = LogisticRegression(class_weight='balanced', n_jobs=-1)
  clf_parameters = {'clf__random_state':(1,10),
                    'clf__solver':('newton-cg','lbfgs'),
                  }

  pipeline = Pipeline([
    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k=2000)),         
    ('clf', clf)])

  feature_parameters = {
    'vect__ngram_range': ((1,1),(1, 2)),  # Unigrams, Bigrams or Trigrams
    }  

  parameters={**feature_parameters,**clf_parameters}

  grid = GridSearchCV(pipeline,parameters,n_jobs=-1,scoring='f1_micro',cv=10)

  grid.fit(X_train, y_train)

  clf = grid.best_estimator_

  print('\nThe best set of parameters of the pipeline are: ')
  print(clf)

  y_pred_lr = clf.predict(X_test)
  
  return y_pred_lr

X_test = tst_df['processed_text'].values

predictions = LogReg(X_train, X_test, y_train, y_test)

predictions_og = []
for i in range(len(predictions)):
    if predictions[i] == 0:
        predictions_og.append('negative')
    elif predictions[i] == 1:
        predictions_og.append('neutral')
    else:
        predictions_og.append('positive')

output = pd.DataFrame(predictions_og)

np.savetxt('Ayush_Yadav_labels.txt', output.values, fmt='%s')