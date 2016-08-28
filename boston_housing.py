# Import libraries necessary for this project
import numpy as np
import pandas as pd
#import visuals as vs # Supplementary code
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
import utils

# Pretty display for notebooks
#% matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)


def show_stats(features, prices, data):
    minimum_price = np.amin(prices)
    maximum_price = np.amax(prices)
    mean_price = np.mean(prices)
    median_price = np.median(prices)
    std_price = np.std(prices)

    # Show the calculated statistics
    print "Statistics for Boston housing dataset:\n"
    print "Minimum price: ${:,.2f}".format(minimum_price)
    print "Maximum price: ${:,.2f}".format(maximum_price)
    print "Mean price: ${:,.2f}".format(mean_price)
    print "Median price ${:,.2f}".format(median_price)
    print "Standard deviation of prices: ${:,.2f}".format(std_price)
    print "\n"

def dtr_default(features_train, features_test, labels_train, labels_test):    
    reg = DecisionTreeRegressor()
    reg.fit(features_train, labels_train)
    pred = reg.predict(features_test)
    utils.printM(pred, labels_test)
    utils.printR2(pred, labels_test)

show_stats(features, prices, None)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
          features, prices, test_size=0.4, random_state=0)
dtr_default(features_train, features_test, labels_train, labels_test)

