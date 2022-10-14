import numpy as np
import os
import cv2
import csv
import json
#from numpy.lib.type_check import real
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy._lib.doccer import doc_replace
# from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
# from sklearn import model_selection
# from sklearn import svm
# from sklearn import preprocessing
from sklearn.metrics import accuracy_score,recall_score
# import joblib
# import datetime
import math
# from sklearn.utils import shuffle
from preprocess_image import ela_fft

decay=0.01
reg_strength = 100000 # regularization strength
learning_rate = 0.1
jpg_quality = 95
ELAscale = 15
FTscale = 250
working_dataset = './working_dataset/Dataset01'
training_folder = './training/svm'
training_batch = '5'
training_epoch = 1000
batch_size=50
result_txt = os.path.join(training_folder, training_batch, 'output.txt')
result_dict = os.path.join(training_folder, training_batch, 'stats.json')

x_train, x_test, y_train, y_test = [], [], [], []
output_dict = {"tp":0, "fp":0, "fn":0, "tn":0}
def calc_stat(the_list,name_string):
    #[TP, TN, FP, FN]
    stat_list={
        "object":name_string,
        "accuracy":"",
        "precision":"",
        "negative predictive value":"",
        "recall rate":"",
        "specificity":"",
        "MCC":"",
        "f1-score":"",
        "tp":the_list[0],
        "tn":the_list[1],
        "fp":the_list[2],
        "fn":the_list[3],
    }
    stat_list["accuracy"]=(the_list[0]+the_list[1])/(sum(value for value in the_list))
    if (the_list[0]+the_list[2])!=0:
        stat_list["precision"]=(the_list[0]/(the_list[0]+the_list[2]))
    else:
        stat_list['precision']='ZeroDivisionError'
    if (the_list[1]+the_list[3])!=0:
        stat_list["negative predictive value"]=(the_list[1]/(the_list[1]+the_list[3]))
    else:
        stat_list['negative predictive value']='ZeroDivisionError'
    if (the_list[0]+the_list[3])!=0:
        stat_list["recall rate"]=(the_list[0]/(the_list[0]+the_list[3]))
    else:
        stat_list["recall rate"]='ZeroDivisionError'
    if (the_list[1]+the_list[2])!=0:
        stat_list["specificity"]=(the_list[1]/(the_list[1]+the_list[2]))
    else:
        stat_list["specificity"]='ZeroDivisionError'
    if ((the_list[0]+the_list[2])*(the_list[0]+the_list[3])*(the_list[1]+the_list[2])*(the_list[1]+the_list[3])!=0):
        try:
            stat_list["MCC"]=((the_list[0]*the_list[1])-(the_list[2]*the_list[3]))/(math.sqrt((the_list[0]+the_list[2])*(the_list[0]+the_list[3])*(the_list[1]+the_list[2])*(the_list[1]+the_list[3])))
        except Exception as error:
            stat_list["MCC"]=error.__class__.__name__
    else:
        stat_list["MCC"]=np.nan
    try:
        stat_list["f1-score"]=2*(stat_list['precision']*stat_list['recall rate']/(stat_list['precision']+stat_list['recall rate']))
    except Exception as error:
        print(error.__class__.__name__)
        stat_list["f1-score"]=error.__class__.__name__
    return stat_list

def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped

def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped

# >> FEATURE SELECTION << #
# >> MODEL TRAINING << #
def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)
    
    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            #print('di',(reg_strength * Y_batch[ind] * X_batch[ind]))
            di = W - (reg_strength * (Y_batch * np.dot(X_batch, W))[ind])
        dw += di

    dw = dw/len(Y_batch)  # average
    W=W-learning_rate*dw
    return W
    
def calculate_cost_gradient_v2(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    
    distance = 1-(Y_batch * np.dot(X_batch, W))
    #print("Y_batch",Y_batch)
    #print("distance.shape:",distance)
    #dw = np.zeros(len(W))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            gradient = W
        else:
            #gradient = W - (reg_strength * Y_batch[ind] * X_batch[ind])
            gradient = W - (reg_strength * distance[ind])
        
        W=W-learning_rate*gradient
    #dw = dw/len(Y_batch)  # average
    
    return W,distance

def sgd(features, outputs):
    global training_epoch
    max_epochs = training_epoch

    weights = np.zeros(features.shape[1])
    nth = 2
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        # X, Y = shuffle(features, outputs)

        ascent = calculate_cost_gradient(weights, features, outputs)
        weights = weights - (learning_rate * ascent)
        # convergence check on 2^nth epoch
        if epoch == 4 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights

def sgd_batch(features, outputs,train_dataset_size,learning_rate,decay,PERIOD):
    max_epochs =10
    weights = np.random.randint(1000,size=(features.shape[1]))
    weights=weights.astype(float)
    nth = 2
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        print('epoch:',epoch)
        # shuffle to prevent repeating update cycles
        idxs = np.random.permutation(train_dataset_size) #shuffled ordering
        X_random = features[idxs]
        Y_random = outputs[idxs]
        num_sum=[]
        num_sum_dist=0
        for i in range(PERIOD):
            
            batch_X = X_random[i * batch_size:(i+1) * batch_size]
            batch_Y = Y_random[i * batch_size:(i+1) * batch_size]
            n=len(batch_X)
            y_predicted=batch_Y * np.dot(batch_X, weights)
            weight_derivative=[]
            #for i in range(n):
                #weight_derivative.append(-(2/n) * sum(batch_X[i] * (batch_Y-y_predicted)))
            #print("weight_derivative",weight_derivative)
            #print("period",i)
            
            weights= calculate_cost_gradient(weights, batch_X, batch_Y)
            
            #print("learning_rate*ascent",learning_rate*ascent)
            #weights = weights - (learning_rate * weight_derivative)
            #print("weights after ascent",weights)
            num_sum.append(np.sum(np.dot(batch_X, weights)))
           # num_sum_dist+=dist
            #weights = weights - (learning_rate * ascent)
            # convergence check on 2^nth epoch
            #if epoch == 2 ** nth or epoch == max_epochs - 1:
            if epoch == 4 * nth or epoch == max_epochs - 1:
                cost = compute_cost(weights, batch_X, batch_Y)
                print("Epoch is:{} and Cost is: {}".format(epoch, cost))
                # stoppage criterion
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    return weights
                prev_cost = cost
                nth += 1
        learning_rate=learning_rate * (1.0 / (1.0 + decay * epoch))
        print(np.median(num_sum))
        #print('distance:',num_sum_dist)
    return weights

def main():
    global x_train, x_test, y_train, y_test
    for dataset_type in os.listdir(working_dataset):
        print('dataset type',dataset_type)
        dataset_type_folder = os.path.join(working_dataset, dataset_type)
        for dataset_class in os.listdir(dataset_type_folder):
            dataset_class_folder = os.path.join(dataset_type_folder, dataset_class)
            for filename in os.listdir(dataset_class_folder):
                
                if os.path.exists(os.path.join(training_folder, training_batch, 'data', dataset_type, dataset_class, filename)):
                    fft_image = cv2.imread(os.path.join(training_folder, training_batch, 'data', dataset_type, dataset_class, filename), cv2.IMREAD_COLOR)
                    fft_image= cv2.resize(fft_image,(200,200))
                    fft_image = fft_image.flatten()
                    
                    if dataset_type == 'train':
                        x_train.append(fft_image)
                        if dataset_class == 'real':
                            y_train.append(1)
                        elif dataset_class == 'fake':
                            y_train.append(0)
                    elif dataset_type == 'test':
                        x_test.append(fft_image)
                        if dataset_class == 'real':
                            y_test.append(1)
                        elif dataset_class == 'fake':
                            y_test.append(0)
                else:
                    if not os.path.exists(os.path.join(training_folder, training_batch, 'data', dataset_type, dataset_class)):
                        os.makedirs(os.path.join(training_folder, training_batch, 'data', dataset_type, dataset_class))
                    filepath = os.path.join(dataset_class_folder, filename)
                    fft_image = ela_fft(cv2.imread(filepath))
                    
                    cv2.imwrite(os.path.join(training_folder, training_batch, 'data', dataset_type, dataset_class, filename), fft_image)

                    fft_image = fft_image.flatten()

                    if dataset_type == 'train':
                        x_train.append(fft_image)
                        if dataset_class == 'real':
                            y_train.append(1)
                        elif dataset_class == 'fake':
                            y_train.append(0)
                    elif dataset_type == 'test':
                        x_test.append(fft_image)
                        if dataset_class == 'real':
                            y_test.append(1)
                        elif dataset_class == 'fake':
                            y_test.append(0)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    x_train = MinMaxScaler().fit_transform(x_train)

    print('Training Model')
    train_dataset_size,_= x_train.shape
    PERIOD = int(train_dataset_size/batch_size) #Number of iterations for each epoch
    W = sgd_batch(x_train, y_train,train_dataset_size,learning_rate,decay,PERIOD)
    print(W)
    print('Finish training Model')

    y_test_predicted = np.array([])

    for i in range(x_test.shape[0]):
        yp = np.sign(np.dot(W, x_test[i])) #model
        y_test_predicted = np.append(y_test_predicted, yp)
    print("y_test_predicted",y_test_predicted)
    accuracy = accuracy_score(y_test, y_test_predicted)
    recall = recall_score(y_test, y_test_predicted)
    precision = recall_score(y_test, y_test_predicted)

    print("accuracy on test dataset: {}".format(accuracy))
    print("recall on test dataset: {}".format(recall))
    print("precision on test dataset: {}".format(precision))

    with open(result_txt, 'w') as f:
        f.writelines([f'Epoch 10\n',f'Accuracy {accuracy}\n', f'Recall {recall}', f'Precision {precision}'])
    np.save(os.path.join(training_folder, training_batch, 'weights'),W)

def test():
    weights = np.load(os.path.join(training_folder, training_batch, 'weights.npy'))
    dataset_folder = f'{training_folder}/{training_batch}/data/test'
    for dataset_class in os.listdir(dataset_folder):
        dataset_class_folder = os.path.join(dataset_folder, dataset_class)
        for filename in os.listdir(dataset_class_folder):
            fft_image = cv2.imread(os.path.join(dataset_class_folder, filename), cv2.IMREAD_COLOR)
            fft_image = fft_image.flatten()
            result = np.sign(np.dot(weights,fft_image))
            print(filename, result)
            
            if dataset_class == 'real':
                if result == 1:
                    output_dict['tn'] += 1
                elif result == 0:
                    output_dict['fn'] += 1
            elif dataset_class == 'fake':
                if result == 1:
                    output_dict['fp'] += 1
                elif result == 0:
                    output_dict['tp'] += 1

    stats_dict = calc_stat([output_dict['tp'], output_dict['tn'], output_dict['fp'], output_dict['fn']], 'Real_Human_Face')
    with open(result_dict, 'w') as f:
        json.dump(stats_dict, f, indent=4)

main()
# test()

# TODO: train again on balanced dataset
# TODO: train again on adam optimizer
