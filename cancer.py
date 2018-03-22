import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from sklearn import linear_model
from sklearn.metrics import r2_score


def get_data():
    #Gets all of the data from the breastcancer file,
    #then throws it into a dictionary, with the patient ID
    # as the key. Then the rest as the value.
    file_name = "breastcancer.txt"
    file_object = open(file_name,"r")
    info = file_object.read()
    info = info.split("\n")
    breastcancer_dict = dict()
    for spot in info:
        total = spot.split(',')
        ID = total[:1][0]
        data = total[1:]
        if data != []:
            breastcancer_dict[ID] = data

    #the key is the ID
    #0-8 are values that can be tested on
    #9 is either 2 or 4 for the badness of the tumor.
    return breastcancer_dict

def plot(data):
    pass

def create_line(slope, intercept):
    """
    Plot a line from slope and intercept
    """
    x = list()
    y = list()

    for i in range(17):
        y.append(intercept + slope * i)
        x.append(i)
    return x,y

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) - mean(xs *ys)))

    b = mean(ys) - m * mean(xs)
    return m,b

def full_linear_reg():
    data_dict = get_data()
    data_fun = selection(data_dict,3,6)

    malgval1,malgval2 = make_post_lst_same(data_fun)
    m,b = best_fit_slope_and_intercept(np.array(malgval1),np.array(malgval2))
    x,y = create_line(m,b)
    print m,b
    regression_line = [(m*x)+b for x in malgval1]
    r_squared = coefficient_of_determination(np.array(malgval2),regression_line)
    print r_squared
    plt.scatter(malgval1,malgval2,color='#003F72', label = 'data')
    plt.plot(malgval1, regression_line, label = 'regression line')
    plt.axis([0, 11, 0, 11])
    plt.show()

def selection(data_dict, spot1,spot2):
    new_dict = dict()
    for key in data_dict:
        #print data_dict[key]
        data1 = data_dict[key][spot1]
        data2 =data_dict[key][spot2]
        val = data_dict[key][9]
        new_dict[key] = [data1,data2,val]
    return new_dict

def make_post_lst_same(data):
    value1 = list()
    value2 = list()
    result = list()
    for key in data:
        total = data[key]
        value1.append(int(total[0]))
        value2.append(int(total[1]))
        #result.append(total[2])

    #the first set to look at
    #the second set of data to look at
    #The result of the tumor
    return value1, value2

def make_plot_lst(data):
    malgval1 = list()
    malgval2 = list()
    boidval1 = list()
    boidval2 = list()
    for key in data:
        total = data[key]
        if(total[2] == '4'):
            malgval1.append(int(total[0]))
            malgval2.append(int(total[1]))
        else:
            boidval1.append(int(total[0]))
            boidval2.append(int(total[1]))
    return malgval1,malgval2,boidval1,boidval2

def squared_error(ys_orig, ys_line):
    return sum((ys_line -ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

def skit():
    regr = linear_model.LinearRegression()
    data_dict = get_data()
    data_fun = selection(data_dict,3,6)
    lst1, lst2 = make_post_lst_same(data_fun)

    lst1 = np.array(lst1, dtype=np.float64)
    lst2 = np.array(lst2, dtype=np.float64)
    lst1 = lst1.reshape(-1, 1)
    lst2 = lst2.reshape(-1, 1)
    # Train the model using the training sets
    regr.fit(lst1,lst2)

    # Make predictions using the testing set
    y_pred = regr.predict(lst1)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error


    # Plot outputs
    plt.scatter(lst1, lst2,  color='black')
    plt.plot(lst1,y_pred, color='blue', linewidth=3)

    plt.axis([0, 11, 0, 11])
    plt.show()
    print regr.coef_

def main():

    skit()
    #linear_regression()
    full_linear_reg()

main()
