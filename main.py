# coding: utf-8
'''
@author: rohangupta
'''

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import random
import pickle
import gzip
from PIL import Image
import os

### Constants
NUM_CLASSES = 10
LENGTH_FEATURES = 784

### Function for loading MNIST Data
def loadMNISTData():
    trainData = []
    trainTarget = []
    validData = []
    validTarget = []
    testData = []
    testTarget = []

    filename = 'data/mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, testing_data = pickle.load(f)
    f.close()
    print("MNIST Data Loaded!")

    for i in range(len(training_data[0])):
        trainData.append(training_data[0][i])
        trainTarget.append(training_data[1][i])

    for i in range(len(validation_data[0])):
        validData.append(validation_data[0][i])
        validTarget.append(validation_data[1][i])

    for i in range(len(testing_data[0])):
        testData.append(testing_data[0][i])
        testTarget.append(testing_data[1][i])

    trainData = np.asmatrix(trainData)
    trainTarget = np.asmatrix(trainTarget).reshape(-1, 1)

    validData = np.asmatrix(validData)
    validTarget = np.asmatrix(validTarget).reshape(-1, 1)

    testData = np.asmatrix(testData)
    testTarget = np.asmatrix(testTarget).reshape(-1, 1)

    print(trainData.shape)
    print(trainTarget.shape)

    print(validData.shape)
    print(validTarget.shape)

    print(testData.shape)
    print(testTarget.shape)

    #trainData = trainData[:1000]
    #trainTarget = trainTarget[:1000]

    return trainData, validData, testData, trainTarget, validTarget, testTarget

### Function for loading USPS Data
def loadUSPSData():
    USPSMat  = []
    USPSTar  = []
    curPath  = 'data/USPSdata/Numerals'
    savedImg = []

    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                savedImg = img
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    print("USPS Data Loaded!")

    USPSMat = np.asmatrix(USPSMat)
    USPSTar = np.asmatrix(USPSTar).reshape(-1, 1)

    print(USPSMat.shape)
    print(USPSTar.shape)

    return USPSMat, USPSTar

### Function for the Adding Bias Term
def addBias(trainData, validData, testData):
    tempTrainData = trainData
    trainData = np.ones((trainData.shape[0], trainData.shape[1]+1))
    trainData[:, :-1] = tempTrainData

    tempValidData = validData
    validData = np.ones((validData.shape[0], validData.shape[1]+1))
    validData[:, :-1] = tempValidData

    tempTestData = testData
    testData = np.ones((testData.shape[0], testData.shape[1]+1))
    testData[:, :-1] = tempTestData
    return trainData, validData, testData

### Function for calculating Output (y)
def calculateLogisticOutput(theta, X):
    y = np.dot(X, theta)
    y = softmax(y)
    return y

### Function for calculating Accuracy using one-hot vector
def calculateLogisticAccuracy(y, t):
    counter = 0
    for i in range(len(y)):
        if (np.argmax(y[i]) == t[i]):
            counter += 1

    acc = float(counter*100)/len(t)
    return acc

###	Function for calculating Softmax
def softmax(z):
    exp = np.exp(z - np.max(z, axis=1).reshape((-1, 1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

###	Function for converting Scalar to One Hot Vector
def oneHotEncode(targets):
	return np.eye(NUM_CLASSES)[targets]  

### Function for creating Confusion Matrix
def createConfusionMatrix(targets, predictedTargets):
    print("Confusion Matrix:")
    confusionMat = confusion_matrix(targets, predictedTargets)
    print(confusionMat)

### Function for calculating Average Precision
def calculateAveragePrecision(targets, predictedTargets):
    y = label_binarize(targets, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = label_binarize(predictedTargets, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    precision = np.zeros(len(y))
    recall = np.zeros(len(y))
    average_precision = np.zeros(len(y))

    for i in range(NUM_CLASSES):
        #precision[i], recall[i], thresholds = precision_recall_curve(y[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y[:, i], y_pred[:, i])

    #precision_micro, recall_micro, thresholds = precision_recall_curve(y.ravel(), y_pred.ravel())
    average_precision_micro = average_precision_score(y, y_pred, average="micro")
    print('Average Precision Score (micro-averaged over all classes): ' + str(average_precision_micro))
    print("\n")

### Function for performing Logistic Regression
def performMulticlassLogisticRegression(trainData, validData, testData, trainTarget, validTarget, testTarget, USPSMat, USPSTar):
    ### Setting Hyperparameters
    EPOCHS = 1500
    ETA = 0.05
    BATCH_SIZE = 256

    trainAcc = []
    validAcc = []
    testAcc = []

    oneHotTarget = oneHotEncode(trainTarget.transpose()).reshape(-1, 10)

    #trainData, validData, testData = addBias(trainData, validData, testData)

    ### Initializing Weights
    theta = np.matrix(np.ones((trainData.shape[1], NUM_CLASSES)))
    theta = 5 * theta
    print("Weights shape: ", theta.shape)

    ### Performing Mini-batch Stochastic Gradient Descent
    for i in tqdm(range(EPOCHS)):

        ### Starting Batch Training
        for start in range(0, len(trainData), BATCH_SIZE):
            end = start + BATCH_SIZE
            z = np.dot(trainData[start:end], theta)
            a = softmax(z)

            gradient = np.dot(np.transpose(np.subtract(a, oneHotTarget[start:end])), trainData[start:end]) / BATCH_SIZE #trainTarget.size
            gradient = gradient.transpose()
            eta_gradient = ETA * gradient
            theta = np.subtract(theta, eta_gradient)

        trainY = calculateLogisticOutput(theta, trainData)
        tempAcc = calculateLogisticAccuracy(trainY, trainTarget)
        trainAcc.append(tempAcc)

        validY = calculateLogisticOutput(theta, validData)
        tempAcc = calculateLogisticAccuracy(validY, validTarget)
        validAcc.append(tempAcc)

        if (i%100 == 0):
        	print("Iteration: " + str(i) + ", Training Acc = " + str(trainAcc[i]) + ", Validation Acc = " + str(validAcc[i]))

    print("=========Results for Logistic=========")

    predictedMNISTTarget = calculateLogisticOutput(theta, testData)
    testAcc = calculateLogisticAccuracy(predictedMNISTTarget, testTarget)
    print("MNIST Testing Accuracy: ", str(testAcc))

    predictedUSPSTarget = calculateLogisticOutput(theta, USPSMat)
    testUSPSAcc = calculateLogisticAccuracy(predictedUSPSTarget, USPSTar)
    print("USPS Testing Accuracy: ", str(testUSPSAcc))

    createConfusionMatrix(testTarget, np.argmax(predictedMNISTTarget, axis=1))
    calculateAveragePrecision(testTarget, np.argmax(predictedMNISTTarget, axis=1))

    createConfusionMatrix(USPSTar, np.argmax(predictedUSPSTarget, axis=1))
    calculateAveragePrecision(USPSTar, np.argmax(predictedUSPSTarget, axis=1))
    
    return predictedMNISTTarget, predictedUSPSTarget


### Initializing the weights to Normal Distribution
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

### Function for calculating Accuracy
def calculateAccuracy(y, t):
    counter = 0
    for i in range(len(y)):
        if (y[i] == t[i]):
            counter += 1

    acc = float(counter*100)/len(t)
    return acc

### Function for performing Neural Networks
def performNeuralNetwork(trainData, validData, testData, trainTarget, validTarget, testTarget, USPSMat, USPSTar):
    # Setting Hyperparameters
    EPOCHS = 400
    LEARNING_RATE = 0.025
    BATCH_SIZE = 256

    trainAcc = []
    validAcc = []
    testAcc = []

    oneHotTrainTarget = oneHotEncode(trainTarget.transpose()).reshape(-1, 10)
    oneHotValidTarget = oneHotEncode(validTarget.transpose()).reshape(-1, 10)

    NUM_HIDDEN_NEURONS_LAYER_1 = 512
    #NUM_HIDDEN_NEURONS_LAYER_2 = 100

    ### Defining Placeholder
    inputTensor  = tf.placeholder(tf.float32, [None, LENGTH_FEATURES])
    outputTensor = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    ### Initializing weights at each layer
    input_hidden_weights  = init_weights([LENGTH_FEATURES, NUM_HIDDEN_NEURONS_LAYER_1])
    #hidden_hidden_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, NUM_HIDDEN_NEURONS_LAYER_2])
    hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, NUM_CLASSES])

    ### Computing values at each layer
    hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))
    #hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer, hidden_hidden_weights))
    output_layer = tf.matmul(hidden_layer, hidden_output_weights)

    ### Defining Error Function
    error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

    ### Defining Learning Algorithm and Training Parameters
    training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

    ### Prediction Function
    prediction = tf.argmax(output_layer, 1) 

    with tf.Session() as sess:
        
        tf.global_variables_initializer().run()
        
        for epoch in tqdm(range(EPOCHS)):

            ### Shuffling the Training Data at each epoch
            p = np.random.permutation(range(len(trainData)))
            trainData  = trainData[p]
            oneHotTrainTarget = oneHotTrainTarget[p]
            
            ### Starting Batch Training
            for start in range(0, len(trainData), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(training, feed_dict={inputTensor: trainData[start:end], 
                                              outputTensor: oneHotTrainTarget[start:end]})

            # Calculating Training Accuracy for an epoch
            trainAcc.append(np.mean(np.argmax(oneHotTrainTarget, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: trainData,
                                                             outputTensor: oneHotTrainTarget})))

            # Calculating Validation Accuracy for an epoch
            validAcc.append(np.mean(np.argmax(oneHotValidTarget, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: validData,
                                                             outputTensor: oneHotValidTarget})))

            if(epoch%100 == 0):
                print("Iteration: " + str(epoch) + ", Training Acc = " + str(trainAcc[epoch]*100) + ", Validation Acc = " + str(validAcc[epoch]*100))

                if epoch > 300:
                    predictedTestTarget = sess.run(prediction, feed_dict={inputTensor: testData})
                    predictedUSPSTarget = sess.run(prediction, feed_dict={inputTensor: USPSMat})

                    accMNIST = calculateAccuracy(predictedTestTarget, testTarget)
                    print("MNIST Testing Accuracy: " + str(np.around(accMNIST, 2)) + "%")

                    accUSPS = calculateAccuracy(predictedUSPSTarget, USPSTar)
                    print("USPS Testing Accuracy: " + str(np.around(accUSPS, 2)) + "%")
            
        # Testing
        predictedMNISTTarget = sess.run(prediction, feed_dict={inputTensor: testData})
        predictedUSPSTarget = sess.run(prediction, feed_dict={inputTensor: USPSMat})

    print("=========Results for Neural=========")
    print("MNIST Training Accuracy: " + str(np.around(trainAcc[EPOCHS-1]*100, 2)) + "%")
    print("MNIST Validation Accuracy: " + str(np.around(validAcc[EPOCHS-1]*100, 2)) + "%")

    accMNIST = calculateAccuracy(predictedMNISTTarget, testTarget)
    print("MNIST Testing Accuracy: " + str(np.around(accMNIST, 2)) + "%")

    accUSPS = calculateAccuracy(predictedUSPSTarget, USPSTar)
    print("USPS Testing Accuracy: " + str(np.around(accUSPS, 2)) + "%")

    createConfusionMatrix(testTarget, predictedMNISTTarget)
    calculateAveragePrecision(testTarget, predictedMNISTTarget)

    createConfusionMatrix(USPSTar, predictedUSPSTarget)
    calculateAveragePrecision(USPSTar, predictedUSPSTarget)
    
    return predictedMNISTTarget, predictedUSPSTarget


### Function for performing Support Vector Machine
def performSVM(trainData, validData, testData, trainTarget, validTarget, testTarget, USPSMat, USPSTar):
    clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto', decision_function_shape='ovr')
    clf.fit(trainData, np.ravel(trainTarget))

    predictedMNISTTarget = clf.predict(testData)
    predictedUSPSTarget = clf.predict(USPSMat)

    print("=========Results for SVM=========")
    print("MNIST Validation Accuracy: " + str(clf.score(validData, validTarget)*100))
    print("MNIST Testing Accuracy: " + str(clf.score(testData, testTarget)*100))
    print("USPS Testing Accuracy: " + str(clf.score(USPSMat, USPSTar)*100))

    createConfusionMatrix(testTarget, predictedMNISTTarget)
    calculateAveragePrecision(testTarget, predictedMNISTTarget)

    createConfusionMatrix(USPSTar, predictedUSPSTarget)
    calculateAveragePrecision(USPSTar, predictedUSPSTarget)

    return predictedMNISTTarget, predictedUSPSTarget

def performRandomForest(trainData, validData, testData, trainTarget, validTarget, testTarget, USPSMat, USPSTar):
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(trainData, np.ravel(trainTarget))

    predictedMNISTTarget = clf.predict(testData)
    predictedUSPSTarget = clf.predict(USPSMat)

    print("=========Results for Random Forest=========")
    print("MNIST Validation Accuracy: " + str(clf.score(validData, validTarget)*100))
    print("MNIST Testing Accuracy: " + str(clf.score(testData, testTarget)*100))
    print("USPS Testing Accuracy: " + str(clf.score(USPSMat, USPSTar)*100))

    createConfusionMatrix(testTarget, predictedMNISTTarget)
    calculateAveragePrecision(testTarget, predictedMNISTTarget)

    createConfusionMatrix(USPSTar, predictedUSPSTarget)
    calculateAveragePrecision(USPSTar, predictedUSPSTarget)

    return predictedMNISTTarget, predictedUSPSTarget

def performVotingClassification(targets, logisticTarget, neuralTarget, svmTarget, randomForestTarget):
    voteMat = np.zeros(10)
    ensembleTarget = []

    for i in range(len(targets)):
        voteMat[np.argmax(logisticTarget[i])] += 1
        voteMat[neuralTarget[i]] += 1
        voteMat[svmTarget[i]] += 1
        voteMat[randomForestTarget[i]] += 1
        maxi = np.argmax(voteMat)

        if(voteMat[maxi] == 1):
            maxi = neuralTarget[i]

        ensembleTarget.append(maxi)
        #print(voteMat, neuralTarget[i], ensembleTarget[i], USPSTar[i])
        voteMat = np.zeros(10)

    print("=========Results for Ensemble (Majority Voting)=========")
    acc = calculateAccuracy(ensembleTarget, targets)
    print("Ensemble Accuracy: " + str(np.around(acc, 2)) + "%")

    createConfusionMatrix(targets, ensembleTarget)
    calculateAveragePrecision(targets, ensembleTarget)
   


trainData, validData, testData, trainTarget, validTarget, testTarget = loadMNISTData()
USPSMat, USPSTar = loadUSPSData()

logisticMNISTTar, logisticUSPSTar = performMulticlassLogisticRegression(trainData, validData, testData, trainTarget, validTarget, testTarget, USPSMat, USPSTar)
neuralMNISTTar, neuralUSPSTar = performNeuralNetwork(trainData, validData, testData, trainTarget, validTarget, testTarget, USPSMat, USPSTar) 
svmMNISTTar, svmUSPSTar = performSVM(trainData, validData, testData, trainTarget, validTarget, testTarget, USPSMat, USPSTar)
randomForestMNISTTar, randomForestUSPSTar = performRandomForest(trainData, validData, testData, trainTarget, validTarget, testTarget, USPSMat, USPSTar)

performVotingClassification(testTarget, logisticMNISTTar, neuralMNISTTar, svmMNISTTar, randomForestMNISTTar)
performVotingClassification(USPSTar, logisticUSPSTar, neuralUSPSTar, svmUSPSTar, randomForestUSPSTar)






























