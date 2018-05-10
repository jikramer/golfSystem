#http://scikit-learn.org/stable/modules/cross_validation.html#obtaining-predictions-by-cross-validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn import linear_model
import sklearn.metrics as sm

from sklearn.linear_model import LinearRegression
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df1 =''

df =''
X =''
Y =''
X_train =''
X_validation =''
Y_train =''
Y_validation =''

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def loadTrainingSet():
    global df1, X, Y, X_train, X_validation, Y_train, Y_validation
    
    ##preprocess..
    #handle nulls, set to NaN
    array = df1.apply(pd.to_numeric, errors='coerce').values

    #set NaN's to mean
    imputer = Imputer()
    transformed_values = imputer.fit_transform(array)
 	
    X = transformed_values[:, 2:3]	#birdies
    Y = transformed_values[:, 1]	#round score

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    print '-----------------------'
    print 'Training set loaded: '
    print '-----------------------'
    print 'X: ', X
    print ''
    print 'Y: ', Y
    print ''
    print 'X_train: ', X_train
    print ''
    print 'Y_train: ', Y_train
    print ''
 

def getIsUnderPar(roundScore, coursePar):
    if roundScore < coursePar:
        return 1
    return 0


def testLinearRegression():
    global df1, X_train, X_validation, Y_train, Y_validation

    lm = LinearRegression()
    lm.fit(X_train, Y_train)
     
    print X_validation
    print ''
    print 'lm.predict(X_validation): ', lm.predict(X_validation)
    print 'Y_validation: ', Y_validation
    print ''
	 
    return str(X_validation)
	
	
def testBayes():

    global df, X, Y, X_train, X_validation, Y_train, Y_validation

    gnb = GaussianNB()

    ##create these arrays from user data whenever we figure out what we want that to be
    fauxX = np.array([[3],[0],[1],[4],[2]])
    fauxY = np.array([[72],[89],[77],[75],[90]])

    #gnb.fit(X, Y)

    #fitting the incoming user data here..
    gnb.fit(fauxX, fauxY)
    predictions = gnb.predict(X_validation)

    print '-----------------------'
    print 'Bayes results: '
    print '-----------------------'
    
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    
    result = accuracy_score(Y_validation, predictions)
    print " result: " + str(result)
    return str(result)
	
 
	
def loadRound():
    global df, df1
    df = pd.read_csv("../shotlinkData/data/Round2018-724/PGATour/Exports/rround.TXT", delimiter = ';', keep_default_na=False)
	
    # add a new UNDERPAR column to the DataFrame
    df['UNDERPAR'] = df.apply(lambda row: getIsUnderPar(row['Round Score'], row['Course Par']), axis=1)

    columns = ['Total Distance(ft) Prox to Hole','Round Score' , 'Birdies', 'Eagles', 'Pars', 'Bogeys', 'Doubles', 'Others', 'UNDERPAR']
    df1 = pd.DataFrame(df, columns=columns)
    #df1 = df1.set_index('Round Score')
    #print 'describe(): ', df1.describe(include='all')
    #print 'head(): ', df1.head()
 
    loadTrainingSet()
#    testLinearRegression()
  	
