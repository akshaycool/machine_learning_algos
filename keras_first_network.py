'''
We can summarize the construction of deep learning models in Keras as follows:

1) Define your model. Create a sequence and add layers.
2) Compile your model. Specify loss functions and optimizers.
3) Fit your model. Execute the model using data.
4) Make predictions. Use the model to generate predictions on new data.
'''

from keras.models import Sequential
from keras.layers import Dense
import numpy
#fix random seed for reproducability
seed = 7
numpy.random.seed(seed)

#loading the dataset of pima indians diabetes record(binary classification)
dataset = numpy.loadtxt("datasets/pima-indians-diabetes.csv",delimiter = ",")
X = dataset[:,0:8]
Y = dataset[:,8]

#Step1 Define model
#Keras models -> sequence of layers ,input_dim for 8 input elements

model = Sequential()
model.add(Dense(12,input_dim=8,init='uniform',activation='relu'))
model.add(Dense(8,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))

#Compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#Fit model
model.fit(X,Y,nb_epoch=150,batch_size=10)

#evaluate the model
scores = model.evaluate(X,Y)
print("%s: %.2f%%"%(model.metrics_names[1],scores[1]*100))

#predicting the results using model.predicting

predictions = model.predict(X)
#round predictions
rounded = [round(x[0]) for x in predictions]
print (rounded)









