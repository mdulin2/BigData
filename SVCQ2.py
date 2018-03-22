from pandas import read_csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = read_csv("poker_fun.txt")
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(columns=['class']), # xs
    df['class'], # ys
    test_size = 0.40
)
print(SVC(decision_function_shape='ovr').fit(x_train, y_train).score(x_test, y_test))
