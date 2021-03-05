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
    orig_data = pd.read_csv(args.path_to_real)
    synth_data = pd.read_csv(args.path_to_synth)
    synth_data = synth_data.replace([-np.Inf, np.inf], np.nan).dropna()
    synth_data = synth_data[:orig_data.shape[0]]
    sra = run_sra(orig_data, synth_data)
    wass_mu, wass_std = run_wasserstein(orig_data, synth_data)
    base_wass_mu, base_wass_std = run_wasserstein(orig_data, orig_data)
    pmse_rat = run_pmse(orig_data, synth_data)
    print("Synthetic Ranking Agreement", sra)
    print("Baseline Wasserstein distance (mean, std dev)", base_wass_mu, base_wass_std)
    print("Wasserstein distance (mean, std dev)", wass_mu, wass_std)
    print("PMSE ratio", pmse_rat)
    
    
def run_pmse(orig_data, synth_data):
    return pmse_ratio(orig_data, synth_data)
    
    
def run_wasserstein(orig_data, synth_data):
    wass = []
    for i in range(50):
        wass.append(wasserstein_randomization(orig_data, synth_data, 100))

    return np.mean(wass), np.std(wass)
    
    
    
def run_sra(orig_data, synth_data):
    columns_to_predict = orig_data.columns
    
    orig_accuracies = []
    synth_accuracies = []
    for column in columns_to_predict:
        X = orig_data.loc[:, orig_data.columns != column]
        y = orig_data.loc[:, column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
	
        fit = LinearRegression().fit(X_train, y_train)
        predicted = fit.predict(X_test)
        orig_accuracies.append(r2_score(y_test, predicted))
	
        X = synth_data.loc[:, synth_data.columns != column]
        y = synth_data.loc[:, column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
	
        fit = LinearRegression().fit(X_train, y_train)
        predicted = fit.predict(X_test)
        synth_accuracies.append(r2_score(y_test, predicted))
    
    return sra(orig_accuracies, synth_accuracies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script to train differentially private normalizigng flows model")
    parser.add_argument('--path_to_real', default='datasets/adult_subset.csv', type=str, help="Path to real data file")
    parser.add_argument('--path_to_synth', default='synth_data/synth_adult.csv', type=str, help="Path to synthetic data file")
    args = parser.parse_args()
    main(args)
	
	