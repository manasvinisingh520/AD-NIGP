import pandas as pd

grn = pd.read_csv("data/GRN/SEAAD/Microglia/Microglia_GRN.tsv", sep='\t', header=None)
print(grn.head())