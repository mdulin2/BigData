from pandas import read_csv, concat
import numpy as np
import warnings
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
#I worked with Jacob Krantz on this so our code will be practically the same.

def load_data():

    #reads in the wine files, then adds a white or red tag to it.
    df_red = read_csv("./winequality/winequality-red.csv")
    df_red.insert(loc=0,column='type',value=[0.0 for i in range(len(df_red))])
    df_white = read_csv("./winequality/winequality-white.csv")
    df_white.insert(loc=0, column='type',value=[1.0 for i in range(len(df_white))])

    # creates a random order so that the training and testing work.
    df_all = concat([df_red, df_white])
    df_all = df_all.sample(frac=1).reset_index(drop=True)
    full_data = df_all.astype(float).values.tolist()

    #defining areas.
    test_size = 0.10
    train_set = {0.0:[], 1.0:[]} # 0 for red, 1 for white
    test_set = {0.0:[], 1.0:[]}
    df_train = full_data[:-int(test_size*len(full_data))]
    df_test = full_data[-int(test_size*len(full_data)):]

    #dictionary with only two values...1 for white and 0 for red
    for i in df_train:
        train_set[i[0]].append(i[1:])

    for i in df_test:
        test_set[i[0]].append(i[1:])

    return train_set, test_set

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

def testmodel(test_set, train_set):
    correct = 0
    total = 0

    for k_val in [55]:
        for group in test_set:
            for data in test_set[group]:
                vote,confidence = k_nearest_neighbors(train_set, data, k=k_val)
                if group == vote:
                    correct += 1
                total += 1
        print('Accuracy:', correct/float(total))


def sklearn_KNN():
    #brings in the data
    train_set, test_set = load_data()
    result_list_train = list()
    attributes_list_train = list()
    attributes_list_test = list()
    result_list_test= list()

    #shifts the data into a readable format for the library
    for key in train_set:
        for entry in train_set[key]:
            attributes_list_train.append(entry)
            result_list_train.append(key)

    for key in test_set:
        for entry in test_set[key]:
            attributes_list_test.append(entry)
            result_list_test.append(key)

    neigh = KNeighborsClassifier(n_neighbors=11)
    neigh.fit(attributes_list_train,result_list_train)

    iteration = 0
    count = 0
    for entry in attributes_list_test:
        predict = neigh.predict([entry])
        if(predict == result_list_test[iteration]):
            count +=1
        iteration+=1

    print count/ float(len(attributes_list_test))

def main():
    df_train, df_test = load_data()
    #sklearn_KNN()
    #sklearn_KNN()
    testmodel(df_train, df_test)

main()
