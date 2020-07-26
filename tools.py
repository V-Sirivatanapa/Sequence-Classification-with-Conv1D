import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.interpolate import CubicSpline

import tensorflow
from tensorflow.python.keras.layers import Conv1D, BatchNormalization, MaxPool1D, GlobalAveragePooling1D, Dropout, Dense


## LOAD DATA ###################################################################
""" receives 1 argument: 
    path of the folder containing the file """
def LOAD_FROM_DRIVE(folder_dir, file_dir):
    data_dir = str(folder_dir) + '/' + str(file_dir)
    data = np.load(data_dir, allow_pickle=True)
    return data



## PREPROCESS ##################################################################
""" receives 1 argument, unequal array
    returns padded array """
def pad_sequence(x):
    arr_like = np.zeros((99, 13))
    arr_like[:x.shape[0], :x.shape[1]] = x
    return np.asarray(arr_like)

""" receives 2 arguments:
    1. reference
    2. lower limit for length
    returns a mask (list) """
def remove_too_short(x, shorter_than=75):
    return list(pd.Series(x).apply(lambda x: len(x)) >= shorter_than)

""" receives 3 arguments
    1. 1d array (numeric)
    2. class dictionary for all classes
    3. ont_hot (bool) if True return one-hot encoded arrays """
def label_vectorize(y, class_dict, one_hot=False):
    result = []
    if one_hot:
        for i in y:
            arr_init = np.zeros(len(class_dict))
            arr_init[class_dict[i]-1] = 1
            result.append(arr_init)
        return np.asarray(result).astype('float64')

    else:
        for i in y:
            result.append(class_dict[i])
        return np.asarray(result).astype('int64')

    
    
## SUPPORT #####################################################################
""" receives labels and returns a dictionary
    containing each unique label and its corresponding unique value """
def get_class_dict(y):
    return {name:enu for enu, name in enumerate(np.unique(y))}

""" receives 2 arguments 
    1. 1d array for classes
    2. type of input (str)
    returns a dictionary of class weights
    (assigning lowest to higest weights according to
    the proportion of each class in the array)
"""
def get_class_weights(y, input_type='categorical'):
    if input_type=='categorical':
        labels = np.unique(y)
        cw = class_weight.compute_class_weight('balanced', labels ,y)
        return {l:round(c,3) for l, c in zip(labels, cw)}
  
    # elif input_type=='one_hot':
    #   labels, counts = np.unique(np.argmax(y, axis=1)+1, return_counts=True)
    #   sort_index = sorted(range(len(counts)), key=lambda x: counts[x])
    #   labels = labels[sort_index]
    #   counts = counts[sort_index][::-1]/sum(counts)
    #   return {l:round(c,3) for l, c in zip(labels, counts)}

    else:
        print('input_type: not correct')
        pass
    
""" receives array of any dimension
    returns shuffle index with length of the input array
"""
def shuffle_index(x):
    p = np.random.permutation(len(x))
    return p
    
def To_word(y_pred, class_dict):
    reverse_class_dict = {v:k for k, v in class_dict.items()}
    pred_words = []
    for i in y_pred:
        pred_words.append(reverse_class_dict[i])
    return np.asarray(pred_words)
    
## DATA-AUGMENTATION FUNCT #####################################################
""" receives 2 arguments
    1. data to be augmented (datapoint-wise)
    2. augmentation method
    returns augmented data """
def Augment(X, method): 
    X_len = len(X)
    result = []
    
    if method == 'mag_warp':
        for e, i in enumerate(X):
            result.append(DA_MagWarp(i))
            # print('\rProcess: Magnitude Warping [', e+1, '/', X_len,']', end='')

    elif method == 'slice':
        for e, i in enumerate(X):
            result.append(pad_sequence(Slicing(i)))
            # print('\rProcess: Window Slicing [', e+1, '/', X_len,']', end='')
            
    elif method == 'time_mask':
        for e, i in enumerate(X):
            result.append(pad_sequence(Time_masker(i)))
            # print('\rProcess: Window Slicing [', e+1, '/', X_len,']', end='')
            
    elif method == 'channel_mask':
        for e, i in enumerate(X):
            result.append(pad_sequence(Channel_masker(i)))
            # print('\rProcess: Window Slicing [', e+1, '/', X_len,']', end='')
    
    else:
        print('specify a method')
        pass
    
    return np.asarray(result)



## FOR DATA-AUGMENTATION ######################################################
""" receives 2d array
    returns sliced (cropped) array
"""
def Slicing(Xi):
    window_size = 90
    start_index = np.random.randint(0, 10)
    return Xi[0+start_index: window_size + start_index, :]

""" receives 2d array
    this function randomly drops rows in the input array
"""
def Time_masker(Xi):
    X = copy.deepcopy(Xi)
    full_len = X.shape[0]
    all_index = np.arange(full_len)
    
    percentage_del = (np.random.randint(5, 11, 1)/100)[0]
    del_proportion = int(round(full_len*percentage_del))
    
    start_index = np.random.randint(0, full_len-(del_proportion-1))
    delete_index = all_index[start_index:start_index+del_proportion]
    
    new_index = np.delete(all_index, delete_index, None)
    
    return X[new_index]

def Channel_masker(Xi):
    X = copy.deepcopy(Xi)
    randint = np.random.randint(0, 13)
    new_arr = np.zeros((X.shape[0]))
    X[:,randint] = np.zeros((X.shape[0]))
    return X

# adopted and modified code from https://github.com/terryum
def GenerateRandomCurves(Xi, sigma=0.2, knot=4, random_param=True):
    if random_param:
        sigma = (np.random.randint(10, 31, 1)/100)[0] #range[0.1,0.3]
        knot = (np.random.randint(4, 9, 1))[0]        #range[4,8]
    xx = (np.ones((Xi.shape[1],1))*(np.arange(0, Xi.shape[0], (Xi.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, Xi.shape[1]))
    x_range = np.arange(Xi.shape[0])
    y_range = np.arange(Xi.shape[1])
    result = []
    for i in y_range:
        result.append(CubicSpline(xx[:,i], yy[:,i])(x_range))
    return np.asarray(result).transpose()

def DA_MagWarp(Xi):
    return Xi * GenerateRandomCurves(Xi)



## BUILD MODEL & TRAINING #####################################################
""" Conv1d neural nerwork that receives
    1. input (shape=99, 13)
    2. label (int) ## use with loss='sparse_categorical_crossentropy'
    Note: already (slightly) fine-tuned!
"""
def define_model():
    input_shape = (99,13)
    output_shape = 35
    activation = 'relu'
    
    input_layer = keras.Input(shape=input_shape)

    h = Conv1D(256, 5, activation=activation, padding='same')(input_layer)
    h = BatchNormalization()(h)

    h = Conv1D(256, 5, activation=activation, padding='same')(h)
    #h = BatchNormalization()(h)
    h = MaxPool1D(3)(h)

    h = Conv1D(512, 5, activation=activation, padding='same')(h)
    #h = BatchNormalization()(h)
    h = Dropout(0.35)(h)

    h = Conv1D(512, 5, activation=activation, padding='same')(h)
    h = GlobalAveragePooling1D()(h)
    h = Dropout(0.5)(h)

    output_layer = Dense(35, activation='softmax')(h)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model



def train_on_synthetic_data(methods):
    # Original data + Magnitude warping (worked well with VGG)
    def generate_synthetic_data(gen_method):
        X = np.concatenate([X_train_, Augment(X_train_, gen_method)])
        y = np.concatenate([y_train_, y_train_])
        return X, y
    
    models, scores, Y_PRED = [], [], []
    
    for i in methods:
        print('> Generating data: ' + i)
        X, y = generate_synthetic_data(i)
        
        # random permutation
        p = shuffle_index(X)
        X, y = X[p], y[p]
        
        # define models
        model = define_model()
        model.compile(optimizer = tensorflow.keras.optimizers.Adam(lr=0.001),
                      loss      = 'sparse_categorical_crossentropy',
                      metrics   = ['acc']
                     )
        # fit models
        print('> Fitting model')
        model_trained = model.fit(X, y,
                           batch_size      = 128,
                           epochs          = 10,
                           verbose         = 0,
                           class_weight    = my_class_weights,
                           validation_data = (X_val_, y_val_))
        
        models.append(model_trained)
        
        # predict
        print('> Predicting')
        y_pred = model.predict(X_test_)
        y_pred = np.argmax(y_pred, axis=1)
        
        # for confusion matrices
        Y_PRED.append(y_pred)
        
        # calculate score
        scores.append(round(accuracy_score(y_test_, y_pred), 5))
        print()
        
    return models, scores, Y_PRED
    
    
    
## VISUALIZATION ##############################################################
""" receives 2 argument: 
    1. label (numeric, str, bool)
    2. figsize (tuple) """
def plot_class_distribution(classes, figsize=(24, 8)):
    x, y = np.unique(classes, return_counts=True)
    x_tick = np.arange(0, len(x), 1)
    # plot settings
    plt.figure(figsize=figsize)
    plt.bar(x_tick, y, align='center', alpha=0.6)
    plt.xticks(x_tick, x, rotation=45)
    plt.xlabel('Classes: ' + str(len(x)), fontsize=16)
    plt.ylabel('Counts',  fontsize=16)
    plt.title('Distribution of Classes', fontsize=22)
    plt.show()
    
def plot_training_history(model, title=''):
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    
    axs[0].plot(model.history['loss'])
    axs[0].plot(model.history['val_loss'])
    axs[0].set_title('model loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'val'], loc='upper right')
    
    axs[1].plot(model.history['acc'])
    axs[1].plot(model.history['val_acc'])
    axs[1].set_title('model acc')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'val'], loc='lower right')
    
    fig.suptitle(title, fontsize=15)
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)
    plt.tight_layout()