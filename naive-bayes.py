import numpy as np

class NaiveBayes():
    
    """
    class is implementation of simple Nive Bayes algorithm 
    basic math beyond - P(A|B) = P(B|A)P(A)/P(B)
    P(B) is not necessary to compare which probability is higher as it would be same for every case and therefore it is not used in the algorithm
    """
    
    instanceCounterDict = {}
    totalInstanceCounter = {}
    prioProbability = 0.0
    
    """
    input: trainingX - numpy array containing training data to be classifier
           trainingY - numpy array containing labels
    """
    
    def __init__(self, trainingX, trainingY):
        self.trainingX = trainingX
        self.trainingY = trainingY
    
    def train(self):
        #below is accounting for P(A) for every possible case
        self.prioProbability = {instance: (self.trainingY == instance).sum()/len(self.trainingY) for instance in set(self.trainingY)}
        #create base to count P(B|A)
        for sample, label in zip(self.trainingX, self.trainingY):
            #count all parts that can occur in the set
            self.totalInstanceCounter[label] = self.totalInstanceCounter.get(label, 0) + len(sample)
            #count all specific instances separately for the labels
            self.instanceCounterDict[label] = {}
            for case in sample:
                self.instanceCounterDict[label][case] = self.instanceCounterDict[label].get(case, 0) + 1

    def classify(self, toBeClassified, laPlace = 0):
        """
        classifies the label for the queried case by finding highest probability item - P(B|A)P(A) is used to compare the probabilities
        input: toBeClassified - numpy array with one case to be classified
               laPlace - number added to prevent 0 probability of event happening (mainly happens if dataset is too small and not all examples have been used to train)
        """
        #counting P(B|A) probability
        probabilitiesDictionary = {}
        for label in self.trainingY:
            probability = 1.0
            #for every part of toBeClassified count the probability of occuring if the given test was correct choice and multiply all of them
            for case in toBeClassified:
                if case in self.instanceCounterDict[label].keys():
                    probability *= self.instanceCounterDict[label][case] + laPlace / self.totalInstanceCounter[label] + laPlace
                else:
                    probability *= laPlace / self.totalInstanceCounter[label] + laPlace
            #P(B|A)P(A)
            probabilitiesDictionary[probability * self.prioProbability[label]] = label
        #find the highest probability case and return it
        highestProbability = max(probabilitiesDictionary.keys())
        #return class name of highest probability class
        return (probabilitiesDictionary[highestProbability])

test = NaiveBayes(np.array([[1,2],[2,3],[1,2],[2,3]]), np.array([1,2,1,2]))
test.train()
print (test.classify(np.array([1,2])))
print (test.classify(np.array([2,3])))
