from preprocess import *
from sklearn import naive_bayes
from helper import get_instance_words,get_all_instance_words
from Semi_EM_NB import Semi_EM_MultinomialNB
from sklearn import metrics



def classify(clf, clf_name, trainX, trainY,trainU, testX, testY):
    print(clf_name)
    print("Fitting....")
    clf.fit(trainX, trainY, trainU)
    print("Done Fitting.....")

    result = clf.predict(testX)
    result = np.array(result, dtype=bool)
    # Calculate accuracy only when test results are provided
    if result.shape == testY.shape:
        correct_predictions = np.count_nonzero(result == testY)
        print("Accuracy: %.6f\n" % (correct_predictions/result.shape[0]))

    return result

# Args: [], "", ""
def filter(training_data_list, training_data_info, training_data_noninfo, trainU_data, test_data):

    # Get Training and Testing Data
    mapping = extract_words_and_add_to_dict(training_data_list)
    reverse_mapping = get_reverse_mapping(mapping)
    training_data0 = get_data([training_data_info], mapping)
    trainY = np.ones(training_data0.shape[0], dtype=int)

    training_data1 = get_data([training_data_noninfo], mapping)
    trainY = np.append(trainY,np.zeros(training_data1.shape[0],dtype=int))

    trainX = np.append(training_data0, training_data1, axis=0)

    testX = get_data([test_data], mapping)
    testY = np.array([])

    trainU = get_data([trainU_data], mapping)

    # Writes the words corresponding to the instances in the helper file
    # get_all_instance_words(reverse_mapping,testX,"helper.txt")
    # get_instance_words(reverse_mapping,testX[1])

    predictions = classify(Semi_EM_MultinomialNB(), "Semi_EM_MultinomialNB", trainX, trainY, trainU, testX, testY)
    informative_reviews = testX[predictions]
    return (informative_reviews, mapping, reverse_mapping)
