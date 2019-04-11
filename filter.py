from preprocess import *
from sklearn import naive_bayes
from helper import get_instance_words,get_all_instance_words
from Semi_EM_NB import Semi_EM_MultinomialNB
from performance_metrics import get_accuracy
from performance_metrics import get_f_measure
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def classify(clf, clf_name, trainX, trainY, testX, testY, trainU):
    print(clf_name)
    print("Fitting....")
    clf.fit(trainX, trainY,trainU)
    print("Done Fitting.....")

    result = clf.predict(trainU)
    #accuracy = get_accuracy(result, testY)
    #print("\naccuracy\n :", accuracy)
    pred_prob = clf.predict_probability(trainU)

    print("Len Pred Prob", len(pred_prob))
    print("Data Len: ",len(trainU))

    pred_prob = pred_prob[:,1]

    result = np.array(result, dtype=bool)
    # Calculate accuracy only when test results are provided
    # if result.shape == testY.shape:
    #     correct_predictions = np.count_nonzero(result == testY)
    #     print("Accuracy: %.6f\n" % (correct_predictions/result.shape[0]))

    return result, pred_prob

# Args: [], "", ""
def fun(clf, clf_name, trainX, trainY,trainU, testX, testY):
    print("entered f_measure fun")
    clf.fit(trainX, trainY, trainU)
    result = clf.predict(testX)
    temp = get_f_measure(result,testY)

    return temp
def fun1(clf, clf_name, trainX, trainY, testX, testY):
    print("entered f_measure fun1")
    clf.fit(trainX, trainY)
    result = clf.predict(testX)
    temp = get_f_measure(result,testY)
    
    return temp
def filter(training_data_list, training_data_info, training_data_noninfo, test_data_info,test_data_noninfo,trainU_data):

    # Get Training and Testing Data
    mapping = extract_words_and_add_to_dict(training_data_list)
    reverse_mapping = get_reverse_mapping(mapping)

    training_data0 = get_data([training_data_info], mapping)
    trainY = np.ones(training_data0.shape[0], dtype=int)

    training_data1 = get_data([training_data_noninfo], mapping)
    trainY = np.append(trainY,np.zeros(training_data1.shape[0],dtype=int))

    trainX = np.append(training_data0, training_data1, axis=0)


    test_data0 = get_data([test_data_info], mapping)
    test_data1 = get_data([test_data_noninfo], mapping)
    testX = np.append(test_data0, test_data1, axis=0)
    
    testY = np.ones(test_data0.shape[0], dtype=int)
    testY = np.append(testY, np.zeros(test_data1.shape[0], dtype=int))

    trainU = get_data([trainU_data], mapping)
    

    # Writes the words corresponding to the instances in the helper file
    # get_all_instance_words(reverse_mapping,testX,"helper.txt")
    # get_instance_words(reverse_mapping,testX[1])

    # predictions, pred_prob = classify(naive_bayes.BernoulliNB(), "BernoulliNB", trainX, trainY, testX, testY)
    predictions,pred_prob = classify(Semi_EM_MultinomialNB(), "Semi_EM_MultinomialNB", trainX, trainY, testX, testY, trainU)
    print("\nInstances sent to filter: %d" % (trainU.shape[0]))
    informative_reviews = trainU[predictions]
    pred_prob_array = pred_prob[predictions]
    print("Useful instances: %d\n" % (informative_reviews.shape[0]))
    return (informative_reviews, mapping, reverse_mapping, pred_prob_array)
