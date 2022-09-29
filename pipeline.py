# Imports

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest,chi2 

from collections import Counter

# Evaluation Functions

def eval(y_test, y_pred):

    print('\n *************** Classification Report ***************  \n')
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    prcn = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return acc, prcn, recall, f1

# Logistic Regression

def LogReg(X_train, X_test, y_train, y_test, num_feats):
    
  clf = LogisticRegression(class_weight='balanced', n_jobs=-1)
  clf_parameters = {'clf__random_state':(1,10,50,100),
                    'clf__solver':('newton-cg','lbfgs','liblinear','saga'),
                  }

  pipeline = Pipeline([('feature_selector', SelectKBest(chi2, k=num_feats)),         
                      ('clf', clf)])

  parameters={**clf_parameters}

  grid = GridSearchCV(pipeline,parameters,n_jobs=-1,scoring='f1_micro',cv=10)

  grid.fit(X_train, y_train)

  clf = grid.best_estimator_

  print('\nThe best set of parameters of the pipeline are: ')
  print(clf)

  y_pred_lr = clf.predict(X_test)
  
  return y_pred_lr

# Multinomial Naive Bayes Classifier

def NaiveBayes(X_train, X_test, y_train, y_test, num_feats):
    
  clf=MultinomialNB(fit_prior=True, class_prior=None)  
  clf_parameters = {
        'clf__alpha':(0,1),
        } 

  pipeline = Pipeline([('feature_selector', SelectKBest(chi2, k=num_feats)),         
                      ('clf', clf)])

  parameters={**clf_parameters}

  grid = GridSearchCV(pipeline,parameters,n_jobs=-1,scoring='f1_micro',cv=10)

  grid.fit(X_train, y_train)

  clf = grid.best_estimator_

  print('\nThe best set of parameters of the pipeline are: ')
  print(clf)

  y_pred_mnb = clf.predict(X_test)

  return y_pred_mnb

# Random Forest Classifier

def RandFrst(X_train, X_test, y_train, y_test, num_feats):

  clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
  clf_parameters = {
              'clf__criterion':('gini', 'entropy'), 
              'clf__max_features':('sqrt', 'log2'),   
              'clf__n_estimators':(10, 30,50,100,200),
              'clf__max_depth':(10,20),
              } 

  pipeline = Pipeline([('feature_selector', SelectKBest(chi2, k=num_feats)),         
                      ('clf', clf)])

  parameters={**clf_parameters}

  grid = GridSearchCV(pipeline,parameters,n_jobs=-1,scoring='f1_micro',cv=10)

  grid.fit(X_train, y_train)

  clf = grid.best_estimator_

  print('\nThe best set of parameters of the pipeline are: ')
  print(clf)

  y_pred_rf = clf.predict(X_test)
 
  return y_pred_rf

# SVM Classifier

def SVM(X_train, X_test, y_train, y_test, num_feats):
  
  clf = svm.SVC(class_weight='balanced')  
  clf_parameters = {
      'clf__C':(0.1,0.5,1,2,10,50,100),
      'clf__kernel': ('linear', 'rbf','poly')
      }
  
  pipeline = Pipeline([('feature_selector', SelectKBest(chi2, k=num_feats)),         
                      ('clf', clf)])

  parameters={**clf_parameters}

  grid = GridSearchCV(pipeline,parameters,n_jobs=-1,scoring='f1_micro',cv=10)

  grid.fit(X_train, y_train)

  clf = grid.best_estimator_

  print('\nThe best set of parameters of the pipeline are: ')
  print(clf)

  y_pred_svm = clf.predict(X_test)

  return y_pred_svm

# Decision Tree Classider

def DescTree(X_train, X_test, y_train, y_test, num_feats):

  clf = DecisionTreeClassifier(random_state=40)
  clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }

  pipeline = Pipeline([('feature_selector', SelectKBest(chi2, k=num_feats)),         
                      ('clf', clf)])

  parameters={**clf_parameters}

  grid = GridSearchCV(pipeline,parameters,n_jobs=-1,scoring='f1_micro',cv=10)

  grid.fit(X_train, y_train)

  clf = grid.best_estimator_

  print('\nThe best set of parameters of the pipeline are: ')
  print(clf)

  y_pred_dtree = clf.predict(X_test)

  return y_pred_dtree