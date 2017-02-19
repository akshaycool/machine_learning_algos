import csv
import random
import numpy as np
import operator

def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        #dividing the dataset to 67-33 tra-test ratio (split var)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

#numpy implementation
def euclideanDistance(instance1,instance2,length):
    distance = 0
    #length is the no of features for the data instance
    for i in range(length):
        tn = np.array(instance1[i])
        tt = np.array(instance2[i])
        #print tn
        #print tt
        distance += np.sum(np.square(tn - tt))
    return np.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        label = neighbors[x][-1]
        if label in classVotes:
            classVotes[label] +=1
        else:
            classVotes[label] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet,predictions):
    '''
    Here we calculate the accuracy of predictions while iterating over the test set
    '''
    print len(testSet)
    print predictions[:5]
    correct = 0
    for i in range(len(testSet)):
        actual_label = testSet[i][-1]
        if actual_label == predictions[i]:
            correct+=1
    return (correct/float(len(testSet)))*100.0


def main():
    #sample flow of knn
    trainingSet ,testSet = [],[]
    loadDataset(filename,split,trainingSet,testSet)
    predictions = []
    # get k nearest neighbours from the test instances
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
if __name__ == '__main__':
    main()