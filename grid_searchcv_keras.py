import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#Function to create model required for KerasClassifier
def create_model(optimizer='rmsprop',init='glorot_uniform'):
    #create model
    model = Sequential()
    model.add(Dense(12,input_dim=8,init=init,activation='relu')) #input layer
    model.add(Dense(8,init=init,activation='relu')) #middle layer 
    model.add(Dense(1,init=init,activation='sigmoid')) #output layer

    #compile model
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

#fix random seed for reproducability
seed = 7
np.random.seed(seed)

dataset = np.loadtxt("datasets/pima-indians-diabetes.csv",delimiter=',')

X = dataset[:,0:8]
Y = dataset[:,8]

#create model
model = KerasClassifier(build_fn=create_model,verbose=0)
#grid search epochs,batch size and optimizer

batch_size = [10,20,30,40,60,80,100]
epochs = [10,50,100]

param_grid = dict(batch_size=batch_size,nb_epoch=epochs)
grid = GridSearchCV(estimator = model,param_grid=param_grid,n_jobs=-1)
grid_result = grid.fit(X,Y)

#summarize results
print ("Best: %f using %s" %(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean,stdev,param in zip(means,stds,params):
    print ("%f (%f) with: %r"%(mean,stdev,param))












