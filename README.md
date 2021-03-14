# Differentially Private Synthetic Data Generation Using Normalizing Flows

This repository contains the code and scripts to train models that generate differentially private synthetic data from original datasets containing confidential and/or sensitive data. The model is based on the Masked Autoregressive Flows (MAF) architecture and Differentially Private Stochastic Gradient Descent (DP-SGD). The code is implemented in Pytorch and uses the [Opacus](https://opacus.ai/) library for Differential Privacy.

### System Requirements

#### Install dependencies

```
$ pip install -r requirements.txt
```

### How to run the code

All the codes related to the basic synthesizer can be found in the [basic_synthesizer](./basic_synthesizer) directory. The directory also contains example notebooks that demonstrate how to generate the synthetic datasets and test the performance metrics for the basic synthesizer (eg: [adult dataset example](./basic_synthesizer/adult_sample.ipynb)).

All the scripts to train the Differentially Private Normalizing Flows model are located in the [normalizing_flows](./normalizing_flows) directory.
Each of the scripts contain various options to run the code. Please run `python <script>.py --help` to see all the options before running the code.

#### Training the Model

```
$ python train.py
```

#### Generating Diagnostic Plots

```
$ python evaluate_diagnostics.py --model_path <model_file_to_use> --data_name <dataset_name>
```

#### Generating Synthetic Datasets

```
$ python generate_synthetic_dataset.py
```

#### Evaluating Performance Metrics

```
$ python metrics/evaluate.py
```

### Results & Figures

#### Model Diagnostics

#### Synthetic Data vs Real Data

The side by side histograms betwen synthetic data vs real data is available in the following [directory](/normalizing_flows/figs/subset_dp_models/histograms_real_vs_synth). Here are some notable columns within the datasets.
POWER-voltage             |  Adult-age          |  PUMS-education
:-------------------------:|:-------------------------:|:-------------------------:
![](normalizing_flows/figs/subset_dp_models/histograms_real_vs_synth/power_2.png)  |  ![](normalizing_flows/figs/subset_dp_models/histograms_real_vs_synth/adult_0.png)  | ![](normalizing_flows/figs/subset_dp_models/histograms_real_vs_synth/pums_2.png)


#### Performance Metrics

| Dataset | Synthetic Ranking Agreement (SRA) | Wasserstein Ratio | PMSE Ratio |
|--------|-------------|--------|-------------|
| Adult | 0.6 | 0.0834 | 1.8946 |
| PUMS | 0.167 | 0.0934 | 0.7844 |
| Power | 0.71| 0.0838 | 0.5838 |


### License
This code is available under the [MIT License](./LICENSE).

### Terms of Use
This code is free to be used by anyone who would like to run DP-MAF experiments. By using the data and source code in this repository, you agree to their respective licenses. 
