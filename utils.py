from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

def printR2 (pred, labels_test):
    r2 = r2_score(pred, labels_test)
    print "R2 Score ", r2

def printM (pred, labels_test): 
    reg_mae = mae(pred, labels_test)
    reg_mse = mse(pred, labels_test)
    print "MAE is ", reg_mae
    print "MSE is ", reg_mse
