# Evaluate the suite of batch sizes from 10 to 100 in the steps of 20

import numpy
from sklearn import datasets, linear_model, cross_validation, grid_search
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

#Function to create model using prima-indiana-diabetes dataset required for KerasClassifier
def create_model():
    #create model
    model = Sequential()
    model.add(Dense(12,input_dim=8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    #compile model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

#fix random seed for reproducability
seed = 7
numpy.random.seed(seed)

#load dataset 
dataset = numpy.loadtxt("datasets/pima-indians-diabetes.csv",delimiter=',')

X = dataset[:,0:8]
Y = dataset[:,8]

# digits = datasets.load_digits()
# X = digits.data[:1000]
# Y = digits.target[:1000]


kf_total = cross_validation.KFold(len(X), n_folds=10, shuffle=True, random_state=4)
for train, test in kf_total:
    print train, '\n', test, '\n\n'


# lr = linear_model.LogisticRegression()
# g= [lr.fit(X[train_indices], Y[train_indices]).score(X[test_indices],Y[test_indices])
# for train_indices, test_indices in kf_total]

# print g

model = KerasClassifier(build_fn=create_model,verbose=1)

#evaluting model against 10-fold cross validation
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results = cross_val_score(model,X,Y,cv=kfold)
print results.mean()











