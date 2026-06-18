# print sample of immune_cells.h5ad from rosmap data
import anndata as ad
import scanpy as sc
import pandas as pd

#print all possible values of cell_type_high_resolution
adata = ad.read_h5ad('data/ROSMAP_Data/Immune_cells_unprocessed.h5ad')

# filter out cell_type_high_resolution categories "CAMs" and "T cells"
# .copy() materializes the AnnData view so assigning adata.X is safe.
keep = ~adata.obs['cell_type_high_resolution'].isin(['CAMs', 'T cells'])
adata = adata[keep].copy()

#print sample of filtered adata
temp = adata.layers['counts']
adata.X = temp

#load mapping file: individual_Cells_across_studies_individualID.csv
mapping = pd.read_csv(
    r'I:\sf2026\data\ROSMAP_Data\individual_Cells_Across_Studies_individualID.csv',
    usecols=['Mathys_Cell_2023', 'individualID'],
    low_memory=False,
)

mapping = mapping.drop_duplicates('Mathys_Cell_2023')
id_map = mapping.set_index('Mathys_Cell_2023')['individualID']
adata.obs['individualID'] = adata.obs.index.map(id_map)

# load clinical metadata: ROSMAP_clinical.csv, and merge with adata.obs (both share individualID column)
donor_metadata = pd.read_csv("I:\sf2026\data\ROSMAP_Data\ROSMAP_clinical.csv")

adata.obs = adata.obs.join(
    donor_metadata.set_index("individualID"),
    on="individualID",
    how="left",
    rsuffix="_metadata"
)

print(adata.obs.head())
print(adata.shape)
print(adata.obs.columns.tolist())

#save adata to h5ad file
adata.write_h5ad('data/ROSMAP_Data/Microglia.h5ad')