import os 
import numpy as np 
import pandas as pd 

trainPath = '/Users/computer/Documents/AlphabetMNIST/train.csv'
testPath = '/Users/computer/Documents/AlphabetMNIST/test.csv'

def getTrainingData(trainPath = trainPath):
	trainData = pd.read_csv(trainPath)
	labels = trainData['label']
	trainData.drop(columns = ['label'],inplace = True)
	images = np.array(trainData)
	targets = np.array(labels)
	del trainData,labels 
	return images/255.0,np.reshape(targets,(len(targets),1))

def getTestingData(testPath = testPath):
	testData = pd.read_csv(testPath)
	labels = testData['label']
	testData.drop(columns = ['label'],inplace = True)
	images = np.array(testData)
	targets = np.array(labels)
	del testData,labels 
	return images/255.0,np.reshape(targets,(len(targets),1))

