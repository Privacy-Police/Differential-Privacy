import pandas as pd

from basic_synthesizer import BasicSynthesizer


dataset = pd.read_csv('adult.csv')

synthesizer = BasicSynthesizer(dataset)

print(synthesizer.sample(10))