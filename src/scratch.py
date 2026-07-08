# print columns of sea-ad data

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the sea-ad astrocyte cell counts
sea_ad = ad.read_h5ad('data/SEA_AD_Data/Astrocytes/Astrocytes.h5ad')
print("done reading")
print(sea_ad.obs.head())
print(sea_ad.X.shape)

"""
cell_type_ontology_term_id
disease_ontology_term_id
self_reported_ethnicity_ontology_term_id
sex_ontology_term_id
Neurotypical reference
Class
Subclass
Supertype
LATE-NC stage
Microinfarct pathology
development_stage_ontology_term_id
disease
sex
self_reported_ethnicity
"""

#print first 5 of columns above in sea_ad.obs
cols = ['Subclass']
print(sea_ad.obs[cols].head())