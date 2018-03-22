from sklearn import svm

#training
atts = [
    [0,0,0,0],
    [0,0,1,0],
    [1,1,0,1],
    [1,0,0,1],
    [0,1,1,0],
    [0,0,1,1],
    [0,0,0,1],
    [1,1,0,0]
]

#results of the training
results = [0,0,0,1,1,1,1,1]

#setting gamma here every once in a while too.
clf = svm.SVC(degree = 1, C= 14)
clf.fit(atts,results)

test = [[1,1,1,1],
    [0,1,0,1],
    [1,1,0,0]
]

#checking for how the training does when tested on
for i in range(8):
    x = clf.predict([atts[i]])
    if(x[0] == results[i]):
        print True,
    else:
        print False,
print
#predicts the testing values
for i in range(len(test)):
    x = clf.predict([test[i]])
    print x,
