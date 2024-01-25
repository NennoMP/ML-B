import numpy as np
import tensorflow as tf
from training.solver import Solver


def averange_prediction_nn(estimator, params, N_epochs, x_dev, x_test, y_dev, y_test, k, x_blind_test=None, optimizer=None):

    prediction_dev = []
    prediction_test = []
    mse_dev = []
    mse_test = []
    if(x_blind_test is not None): # CUP
        prediction_blind_test=[]
        mee_dev = []
        mee_test = []
    else: # MONK
        acc_dev = []
        acc_test = [] 

    for i in range(k):
        if(optimizer is None): model = estimator(params)
        else: model = estimator(optimizer, params)
        
        if(x_blind_test is not None): # CUP
            solver = Solver(model, x_dev, y_dev, x_test, y_test, target='mean_euclidean_error')
            solver.train(epochs=N_epochs, patience=100)
            pred_dev = model.predict(x_dev)
            pred_test = model.predict(x_test) 
            prediction_blind_test.append(model.predict(x_blind_test))
            mse_e_dev, mee_cup_dev = model.evaluate(x_dev, y_dev)
            mse_e_test, mee_cup_test = model.evaluate(x_test, y_test)
            mee_dev.append(mee_cup_dev)
            mee_test.append(mee_cup_test)
               
        else: # MONK
            solver = Solver(model, x_dev, y_dev, x_test, y_test, target='accuracy')
            solver.train(epochs=N_epochs, patience=50, batch_size=len(x_dev))
            pred_dev = (np.rint(model.predict(x_dev))).astype(int)
            pred_test = (np.rint(model.predict(x_test))).astype(int)
            mse_e_dev, acc_monk_dev = model.evaluate(x_dev, y_dev)
            mse_e_test, acc_monk_test = model.evaluate(x_test, y_test)
            acc_dev.append(acc_monk_dev)
            acc_test.append(acc_monk_test)  

        prediction_dev.append(pred_dev)
        prediction_test.append(pred_test)
        mse_dev.append(mse_e_dev)
        mse_test.append(mse_e_test)        
         
    prediction_dev = np.mean(np.array(prediction_dev), axis=0)
    prediction_test = np.mean(np.array(prediction_test), axis=0)
    mse_mean_dev = np.mean(np.array(mse_dev))
    mse_mean_test = np.mean(np.array(mse_test))
    mse_std_dev = np.std(np.array(mse_dev))
    mse_std_test = np.std(np.array(mse_test))

    
    if(x_blind_test is not None): # CUP
        prediction_blind_test = np.mean(np.array(prediction_blind_test), axis=0)
        mee_mean_dev = np.mean(np.array(mee_dev))
        mee_mean_test = np.mean(np.array(mee_test))
        mee_std_dev = np.std(np.array(mee_dev))
        mee_std_test = np.std(np.array(mee_test))        
        return prediction_dev, prediction_test, prediction_blind_test, mse_mean_dev, mse_mean_test, mse_std_dev, mse_std_test, mee_mean_dev, mee_mean_test, mee_std_dev, mee_std_test

    else: # MONK
        acc_mean_dev = np.mean(np.array(acc_dev))
        acc_mean_test = np.mean(np.array(acc_test))
        acc_std_dev = np.std(np.array(acc_dev))
        acc_std_test = np.std(np.array(acc_test))  
        return prediction_dev, prediction_test, mse_mean_dev, mse_mean_test, mse_std_dev, mse_std_test, acc_mean_dev, acc_mean_test, acc_std_dev, acc_std_test

        
