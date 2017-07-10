import pandas
import numpy as np
from sklearn.utils import shuffle
from keras.utils import np_utils

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

DATA_PATH = 'data/training.txt'
CLASS_PATH = 'data/training_class.txt'

# Parameters
train_ratio = 0.8
patch_size = (1,1)
features = 10
dates = 23



def load_data(reshape=True, one_hot_encode=True):   
    # Load files
    # Create Numpy array

    X = pandas.read_csv(DATA_PATH, header=None).as_matrix()
    y = pandas.read_csv(CLASS_PATH, header=None).as_matrix() - 1

    X, y = shuffle(X, y, random_state=seed)

    print('Values in training set:')
    uniques, ids = np.unique(y, return_inverse=True)
    print(uniques)
    
    if reshape:
        X = X.reshape([X.shape[0], patch_size[0], patch_size[1], features * dates])
        
    if one_hot_encode:
        # One hot encode outputs
        y = np_utils.to_categorical(y)
        
    # Split into training and test data

    n_train = int(train_ratio * X.shape[0])
    X_train, X_test = X[:n_train,:], X[n_train:,:] 
    y_train, y_test = y[:n_train], y[n_train:]

    
    return (X_train, y_train), (X_test, y_test)

