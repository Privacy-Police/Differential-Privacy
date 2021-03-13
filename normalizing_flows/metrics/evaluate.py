import argparse

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sra import sra
from wasserstein import wasserstein_randomization
from pmse import pmse_ratio

def main(args):
    # read the real and synthetic datasets
    orig_data = pd.read_csv(args.path_to_real)
    synth_data = pd.read_csv(args.path_to_synth)
    synth_data = synth_data.replace([-np.Inf, np.inf], np.nan).dropna()
    synth_data = synth_data[:orig_data.shape[0]]
    orig_data_norm = (orig_data- orig_data.mean())/orig_data.std()
    synth_data_norm = (synth_data- synth_data.mean())/synth_data.std()
    
    # call the functions to compute the evaluation metrics 
    sra = run_sra(orig_data_norm, synth_data_norm)
    wass_mu, wass_std = run_wasserstein(orig_data_norm, synth_data_norm)
    base_wass_mu, base_wass_std = run_wasserstein(orig_data_norm, orig_data_norm)
    pmse_rat = run_pmse(orig_data_norm, synth_data_norm)
    print("Synthetic Ranking Agreement", sra)
    print("Baseline Wasserstein distance (mean, std dev)", base_wass_mu, base_wass_std)
    print("Wasserstein distance (mean, std dev)", wass_mu, wass_std)
    print("PMSE ratio", pmse_rat)
    
    
def run_pmse(orig_data, synth_data):
    return pmse_ratio(orig_data, synth_data)
    

def run_wasserstein(orig_data, synth_data):
    # run wassertein test multiple times to find the mean and stddev
    # this is done because the wasserstein randomization ratio is very volatile
    wass = []
    for i in range(50):
        wass.append(wasserstein_randomization(orig_data, synth_data, 100))

    return np.mean(wass), np.std(wass)
    
    
    
def run_sra(orig_data, synth_data):
    columns_to_predict = orig_data.columns
    
    orig_accuracies = []
    synth_accuracies = []
    
    # Train a linear regression model for each attribute in the dataset
    for column in columns_to_predict:
    
        # train linear regression and store accuracy score for the original dataset
        X = orig_data.loc[:, orig_data.columns != column]
        y = orig_data.loc[:, column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
            
        fit = LinearRegression().fit(X_train, y_train)
        predicted = fit.predict(X_test)
        orig_accuracies.append(r2_score(y_test, predicted))
	
        # train linear regression and store accuracy score for the synthetic dataset
        X = synth_data.loc[:, synth_data.columns != column]
        y = synth_data.loc[:, column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
	
        fit = LinearRegression().fit(X_train, y_train)
        predicted = fit.predict(X_test)
        synth_accuracies.append(r2_score(y_test, predicted))
        
    # compute sra based on the accuracy score for all the models trained
    return sra(orig_accuracies, synth_accuracies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script to train differentially private normalizigng flows model")
    parser.add_argument('--path_to_real', default='datasets/adult_subset.csv', type=str, help="Path to real data file")
    parser.add_argument('--path_to_synth', default='synth_data/synth_adult.csv', type=str, help="Path to synthetic data file")
    args = parser.parse_args()
    main(args)
	
	