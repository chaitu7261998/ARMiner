# Naive Bayes classifier

if __name__ == "__main__" :

Map = extract_words_and_add_to(["train.txt"])

train_data_info = get_data(["train_info.txt"],Map)
train_data_non_info = get_data(["train_data_non_info.txt"],Map)

Y = np.ones(train_data_info.shape[0], dtype = int)
Y = np.append(Y, np.zeros(train_data_non_info.shape[0], dtype = int))

train_data = np.append(train_data_info, train_data_non_info, axis = 0)

test_data = get_data(["test.txt"], Map)

classifier = naive_bayes.BernouliNB()

classifier.fit(train_data, test_data)
