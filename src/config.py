"""
Configuration constants for single-cell data processing.
"""

# Mapping of region names to Excel sheet tab indices (0-based)
REGION_TO_TAB = {
    "EC": 0,
    "ITG": 1,
    "PFC": 2,
    "V2": 3,
    "V1": 4,
}

# Valid region choices
REGIONS = list(REGION_TO_TAB.keys())

# Default file naming patterns
ASTROCYTES_BASE_PREFIX = "2025-10-22_Astrocytes_{region}"
DEFAULT_METADATA_PATH = "data/AtlasData/2025-11-16_Astrocytes_metadata.xlsx"

# Date prefixes for file naming
METADATA_DATE_PREFIX = "2025-11-16"
MATRIX_DATE_PREFIX = "2025-10-22"

# Directory paths (relative to project root)
DATA_DIR = "data/AtlasData"
GENE_ANNOTATIONS_FILE = "data/gene_annotations.pkl"
DGE_RESULTS_BASE_DIR = "results"

ROSMAP_DATA_DIR = "data/ROSMAP_Data"
ROSMAP_CELL_TYPES = {"Astrocytes", "Microglia"}
ROSMAP_CONDITION_COL = "braaksc" # 0: 1, 1,2: 2, 3,4: 3, 5,6: 4
ROSMAP_PATIENT_COL = "individualID"

## EC genes from paper
GENES_FROM_PAPER = {}
GENES_FROM_PAPER['EC'] = ['APP', 'APOE', 'SLC1A2', 'MAPT', 'RORB', 'AGBL4',
 'AL392086.3', 'ANKS1B', 'DPYSL3', 'ECHDC2', 'GARNL3', 'GMPR', 'GSN', 'KCNN3', 
 'MAP2', 'MTM1', 'PRMT2', 'RPS6KA2', 'SREBF1', 'TMEM132B', 'TMEM132C', 'TOGARAM2', 
 'ZRANB2', 'ABHD2', 'ACACA', 'AHCYL2', 'CAMKMT', 'CEP85L', 'CHN1', 'CHST11', 'CLEC16A',
  'EPN2', 'FAF1', 'FNDC3B', 'FOXO3', 'GPD2', 'HS2ST1', 'LARGE1', 'LINC00472', 'MAP3K5', 
  'MCC', 'NIPSNAP2', 'NOL4', 'OSBPL6', 'PCSK5', 'PDZRN3', 'PRKD1', 'PTAR1', 'PTN', 'RTN4', 
  'SEC14L1', 'SH3RF1', 'SLC24A3', 'SLC35F1', 'SLC6A1', 'TLK1', 'VTI1A', 'ZNF462', 'ZNF532', 
  'ZNF827', 'GFAP', 'AQP4', 'TNC', 'MAPT', 'MAOB', 'GPC6', 'CRYAB', 'CD44', 'AQP1', 
  'ADAMTSL3', 'GRIA1', 'SLC38A1', 'SPARC', 'GJA1', 'GLUL', 'GRM3', 'MERTK', 'NRXN1', 'SLC1A2']
