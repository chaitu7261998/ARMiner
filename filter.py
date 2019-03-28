from preprocess import *
from sklearn import naive_bayes

def classify(clf, clf_name, trainX, trainY, testX, testY):
    print(clf_name)
    print("Fitting....")
    clf.fit(trainX, trainY)
    print("Done Fitting.....")

    result = clf.predict(testX)

    correct_predictions = np.count_nonzero(result == testY)
    print("Accuracy: %.6f\n" % (correct_predictions/result.shape[0]))


if __name__ == "__main__":

    # Get Training and Testing Data
    mapping = extract_words_and_add_to_dict(["datasets/swiftkey/trainL/info.txt","datasets/swiftkey/trainL/non-info.txt"])
    training_data0 = get_data(["datasets/swiftkey/trainL/info.txt"], mapping)
    trainY = np.ones(training_data0.shape[0], dtype=int)

    training_data1 = get_data(["datasets/swiftkey/trainL/non-info.txt"],mapping)
    trainY = np.append(trainY,np.zeros(training_data1.shape[0],dtype=int))

    trainX = np.append(training_data0, training_data1, axis=0)

    testX = get_data(["datasets/swiftkey/trainL/non-info.txt"], mapping)
    testY = np.zeros(testX.shape[0])

    classify(naive_bayes.BernoulliNB(), "BernoulliNB", trainX, trainY, testX, testY)
    classify(naive_bayes.GaussianNB(), "GaussianNB", trainX, trainY, testX, testY)
    classify(naive_bayes.MultinomialNB(), "MultinomialNB", trainX, trainY, testX, testY)
