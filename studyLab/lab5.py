# -*- coding: utf-8 -*- python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc

def getData():
    #http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
    df = pd.read_csv(r'./database/wdbc.data', header=None)
    print('header', df.head())
    print(df.shape)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    print(le.transform(['M', 'B']))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)
    return X_train, X_test, y_train, y_test

def pipeLineDecisionTreeDesign(X_train, y_train):
    pipe_dt = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=1))
    param_grid=[{'decisiontreeclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None]}]
    gs = GridSearchCV(estimator=pipe_dt,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=2)
    kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        gs.fit(X_train[train], y_train[train])
        score = gs.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    return pipe_dt
    
def pipeLineTrain(pipe_dt, X_train, y_train,X_test,y_test):
    # # Looking at different performance evaluation metrics
    # # Reading a confusion matrix
    pipe_dt.fit(X_train, y_train)
    y_pred = pipe_dt.predict(X_test)
    
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
    
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    #plt.savefig('06_09.png', dpi=300)
    plt.show()
    
def piplinePCA(X_train,y_train):
    pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2),
                        LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000,
                                           random_state=1,C=100.0))
    X_train2 = X_train[:, [4, 14]]
    cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train, y_train))
    fig = plt.figure(figsize=(7, 5))
    #
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train, test) in enumerate(cv):
        probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
        fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
    #
    plt.plot([0, 1],[0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
    #
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')
    #
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    #plt.savefig('06_10.png', dpi=300)
    plt.show()

def pipeLineMLPDesign(X_train, y_train):
    pipe_dt = make_pipeline(StandardScaler(), MLPClassifier(activation='logistic', learning_rate_init=0.1, solver='sgd', alpha=1e-5, random_state=1)) #hidden_layer_sizes=(2,2) 
    
    totalNeurons=10
    arrange=[]
    for i in range(totalNeurons-1):
        arrange.append((i+1,totalNeurons-i-1))
    #print(arrange)

    #param_grid=[{'mlpclassifier__hidden_layer_sizes': [(1,1),(5,5)]}]
    #param_grid=[{'mlpclassifier__learning_rate_init': [0.1,0.01,0.001]}]
    param_grid=[{'mlpclassifier__hidden_layer_sizes': arrange}]
    gs = GridSearchCV(estimator=pipe_dt,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=2)
    #print(pipe_dt)
    kfold = StratifiedKFold(n_splits=5, random_state=1).split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        #print('train=',train)
        #print('test=',test)
        gs.fit(X_train[train], y_train[train])
        score = gs.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
    scoresCV = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)), end='')
    print(' CV accuracy: %.3f +/- %.3f' % (np.mean(scoresCV), np.std(scoresCV)))
    print(gs.cv_results_)
    return pipe_dt

def main():
    X_train, X_test, y_train, y_test = getData()
    #pipe_dt = pipeLineDecisionTreeDesign(X_train, y_train)
    #pipeLineTrain(pipe_dt,X_train, y_train,X_test,y_test)
    
    #piplinePCA(X_train,y_train)
    
    pipe_dt = pipeLineMLPDesign(X_train, y_train)
    pipeLineTrain(pipe_dt,X_train, y_train,X_test,y_test)
    
if __name__ == "__main__":
    main()
