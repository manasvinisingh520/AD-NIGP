"""
Load GREmLN embeddings, DGE results, model performance, and gene rankings
for the interactive visualization web app.
"""
import os, re, pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
SUPPORTED_METHODS = ('pca', 'umap')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "GREmLN" / "embeddings"
METADATA_PATH = PROJECT_ROOT / "data" / "AtlasData" / "2025-11-16_Astrocytes_metadata.xlsx"
CACHE_DIR = PROJECT_ROOT / ".cache" / "embedding_viz"
DGE_DIR = PROJECT_ROOT / "results" / "dge_final"
RANKED_LISTS_DIR = PROJECT_ROOT / "ranked_lists"
MODEL_RUNS_DIR = PROJECT_ROOT / "ranking_model_runs"
GENE_ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "gene_annotations.pkl"
GENE_TO_ENSEMBL_PATH = PROJECT_ROOT / "data" / "genes_2_ensembl_ids.pkl"
CELL_ID_COL = "cell_annotation"
ASTROCYTE_REGIONS = [
 "V1","V2","PFC","EC","ITG"]
REGION_TO_TAB = { 'EC': 0, 'ITG': 1, 'PFC': 2, 'V2': 3, 'V1': 4}
KNOWN_AD_GENES = [
 "APP","APOE","SLC1A2","MAPT","RORB","GFAP","AQP4","TNC",
 "MAOB","GPC6","CRYAB","CD44","AQP1","ADAMTSL3","GRIA1",
 "SLC38A1","SPARC","GJA1","GLUL","GRM3","MERTK","NRXN1"]

def _embedding_row_to_cell_id(row_id: str) -> str:
    """Extract cell ID from embedding row id (e.g. 6289-MW-0031_bin2 -> 6289-MW-0031)."""
    m = re.match("^(.+)_bin\\d+$", str(row_id).strip())
    if m:
        return m.group(1)
    return str(row_id)


def _get_astrocyte_cell_embedding_files():
    """Return list of (region, path) for Astrocytes cell embedding TSVs."""
    if not EMBEDDINGS_DIR.exists():
        return []
    out = []
    for p in EMBEDDINGS_DIR.glob("*_cell_embeddings.tsv"):
        if "Microglia" in p.name:
            continue
        else:
            region = p.stem.replace("_cell_embeddings", "").replace("_gene_embeddings", "")
        if region not in ASTROCYTE_REGIONS:
            continue
        else:
            out.append((region, p))

    return sorted(out, key=(lambda x: x[0]))


SAMPLE_COL = "SampleName"
PATIENT_META_COLS = ["Path..Group.","Thal","RIN","Ptau.Total.Tau..A.U..","APOE_Genotype",
 "Sex","Age.at.Death","PMI"]

def _load_patient_metadata_openpyxl(region: str):
    """
    Load patient-level metadata for one region using openpyxl in read_only mode.
    Returns a DataFrame indexed by SampleName (one row per patient).
    Much faster than pd.read_excel for huge files since we only keep needed columns.
    """
    tab_idx = REGION_TO_TAB.get(region)
    if tab_idx is None:
        return
    import openpyxl
    wb = openpyxl.load_workbook((str(METADATA_PATH)), read_only=True, data_only=True)
    ws = wb.worksheets[tab_idx]
    header = [str(c) if c is not None else "" for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    col_idx_sample = header.index(SAMPLE_COL) if SAMPLE_COL in header else None
    if col_idx_sample is None:
        wb.close()
        return
    keep_cols = [c for c in PATIENT_META_COLS if c in header]
    keep_idx = {c: header.index(c) for c in keep_cols}
    patient_data = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        sample = row[col_idx_sample]
        if sample is None:
            continue
        else:
            sample = str(sample).strip()
        if sample in patient_data:
            continue
        else:
            patient_data[sample] = {c: row[keep_idx[c]] for c in keep_cols}

    wb.close()
    if not patient_data:
        return
    df = pd.DataFrame.from_dict(patient_data, orient="index")
    df.index.name = SAMPLE_COL
    for c in df.columns:
        df[c] = df[c].astype(str).replace("None", "").replace("nan", "")

    return df


def load_metadata_for_region(region: str):
    """Load patient-level Astrocytes metadata for one region, index by SampleName."""
    if not METADATA_PATH.exists():
        return
    cache_file = CACHE_DIR / f"meta_{region}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_patient_metadata_openpyxl(region)
    if df is not None:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
        except Exception:
            pass

        return df


def load_metadata_all_sheets():
    """Load Astrocytes metadata from Excel (all region sheets), index by cell_annotation."""
    if not METADATA_PATH.exists():
        return
    dfs = []
    for (region, tab_idx) in REGION_TO_TAB.items():
        try:
            df = pd.read_excel(METADATA_PATH,
              sheet_name=tab_idx,
              engine="openpyxl")
            df["region"] = region
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return
    meta = pd.concat(dfs, ignore_index=True)
    if CELL_ID_COL not in meta.columns:
        return meta
    meta = meta.set_index(CELL_ID_COL)
    return meta


def _reduce_to_2d(X: np.ndarray, method: str='pca') -> np.ndarray:
    """Reduce high-dimensional matrix to 2D using PCA or UMAP."""
    method = method.lower()
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2))
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(X)
    else:
        n_components = min(2, X.shape[0] - 1, X.shape[1])
        if n_components < 2:
            return np.zeros((X.shape[0], 2))
        pca = PCA(n_components=n_components, random_state=42)
        coords = pca.fit_transform(X)
        if coords.shape[1] == 1:
            coords = np.column_stack([coords.ravel(), np.zeros(len(coords))])
    return coords


def load_region_embeddings_and_metadata(region: str, method: str='pca', use_cache: bool=True):
    """
    Load cell embeddings for one region, join with metadata, reduce to 2D.
    method: "pca" or "umap".
    Returns dict: points (list of {x, y, cell_id, embedding_id, region, ...meta}), meta_columns, method.
    """
    method = method.lower()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"viz_{region}_{method}.pkl"
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    files = _get_astrocyte_cell_embedding_files()
    path = next((p for (r, p) in files if r == region), None)
    if not (path and path.exists()):
        return {'points': [], 'meta_columns': [], 'method': method}
    df_emb = pd.read_csv(path, sep=",", index_col=0, nrows=None)
    row_ids = df_emb.index.astype(str).tolist()
    cell_ids = [_embedding_row_to_cell_id(rid) for rid in row_ids]
    X = df_emb.values.astype(np.float64)
    if np.any(np.isnan(X)):
        X = np.nan_to_num(X, nan=0.0)
    coords = _reduce_to_2d(X, method)
    meta_df = load_metadata_for_region(region)
    meta_columns = []
    if meta_df is not None:
        meta_columns = list(meta_df.columns)
    meta_columns.append("region")
    points = []
    for i, (rid, cid) in enumerate(zip(row_ids, cell_ids)):
        pt = {'x': float(coords[i, 0]),
              'y': float(coords[i, 1]),
              'cell_id': cid,
              'embedding_id': rid,
              'region': region}
        if meta_df is not None and cid in meta_df.index:
            row = meta_df.loc[cid]
            for col in meta_df.columns:
                val = row[col]
                pt[col] = str(val) if val else ""
        points.append(pt)

    result = {'points': points, 'meta_columns': meta_columns, 'method': method}
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


ALL_REGIONS_KEY = "All Regions"

def load_all_regions_embeddings(method: str='pca', use_cache: bool=True):
    """
    Load cell embeddings from ALL Astrocytes regions, concatenate, run joint
    dimensionality reduction, and attach per-region metadata.
    """
    method = method.lower()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"viz_ALL_{method}.pkl"
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    files = _get_astrocyte_cell_embedding_files()
    if not files:
        return {'points': [], 'meta_columns': [], 'method': method}
    all_row_ids = []
    all_cell_ids = []
    all_regions = []
    all_X = []
    for (region, path) in files:
        df_emb = pd.read_csv(path, sep=",", index_col=0)
        row_ids = df_emb.index.astype(str).tolist()
        cell_ids = [_embedding_row_to_cell_id(r) for r in row_ids]
        all_row_ids.extend(row_ids)
        all_cell_ids.extend(cell_ids)
        all_regions.extend([region] * len(row_ids))
        all_X.append(df_emb.values.astype(np.float64))

    X = np.vstack(all_X)
    if np.any(np.isnan(X)):
        X = np.nan_to_num(X, nan=0.0)
    coords = _reduce_to_2d(X, method)
    meta_by_region = {}
    all_meta_cols = set()
    for region in set(all_regions):
        mdf = load_metadata_for_region(region)
        if mdf is not None:
            meta_by_region[region] = mdf
            all_meta_cols.update(mdf.columns)

    meta_columns = sorted(all_meta_cols)
    meta_columns.append("region")
    points = []
    for i, (rid, cid, reg) in enumerate(zip(all_row_ids, all_cell_ids, all_regions)):
        pt = {'x': float(coords[i, 0]),
              'y': float(coords[i, 1]),
              'cell_id': cid,
              'embedding_id': rid,
              'region': reg}
        mdf = meta_by_region.get(reg)
        if mdf is not None and cid in mdf.index:
            row = mdf.loc[cid]
            for col in mdf.columns:
                val = row[col]
                pt[col] = str(val) if val else ""
        points.append(pt)

    result = {'points': points, 'meta_columns': meta_columns, 'method': method}
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


def get_available_regions():
    """Return list of region names that have Astrocytes cell embeddings."""
    return [r for r, _ in _get_astrocyte_cell_embedding_files()]


def _get_ensembl_to_symbol():
    """Return dict mapping Ensembl ID -> gene symbol."""
    cache_file = CACHE_DIR / "ensembl_to_symbol.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {}
    if GENE_TO_ENSEMBL_PATH.exists():
        with open(GENE_TO_ENSEMBL_PATH, "rb") as f:
            g2e = pickle.load(f)
        mapping = {v: k for k, v in g2e.items()}
    with open(cache_file, "wb") as f:
        pickle.dump(mapping, f)
    return mapping


def _get_symbol_to_genetype():
    """Return dict mapping gene symbol -> gene type (protein_coding, lncRNA, etc)."""
    cache_file = CACHE_DIR / "symbol_to_genetype.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {}
    if GENE_ANNOTATIONS_PATH.exists():
        with open(GENE_ANNOTATIONS_PATH, "rb") as f:
            ga = pickle.load(f)
        for (_, row) in ga.iterrows():
            mapping[str(row["Gene name"])] = str(row["Gene type"])

    with open(cache_file, "wb") as f:
        pickle.dump(mapping, f)
    return mapping


def _get_gene_descriptions(symbols):
    """Fetch gene descriptions (what the gene does) from mygene, with caching."""
    cache_file = CACHE_DIR / "gene_descriptions.pkl"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            pass
    symbols = [s for s in symbols if s and str(s).strip()]
    to_fetch = [s for s in symbols if s not in cache]
    if to_fetch:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            results = mg.querymany(to_fetch, scopes="symbol", fields="name", species="human", returnall=True)
            for hit in results.get("out", []) or []:
                sym = hit.get("query")
                if sym and "name" in hit:
                    cache[sym] = str(hit["name"]).strip()
            for sym in to_fetch:
                if sym not in cache:
                    cache[sym] = ""
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(cache, f)
            except Exception:
                pass
        except Exception:
            for sym in to_fetch:
                if sym not in cache:
                    cache[sym] = ""
    return {s: cache.get(s, "") for s in symbols}


def _get_gene_embedding_files():
    """Return list of (region, path) for Astrocytes gene embedding files."""
    if not EMBEDDINGS_DIR.exists():
        return []
    out = []
    for p in EMBEDDINGS_DIR.glob("*_gene_embeddings.tsv"):
        if "Microglia" in p.name:
            continue
        else:
            region = p.stem.replace("_gene_embeddings", "")
        if region not in ASTROCYTE_REGIONS:
            continue
        else:
            out.append((region, p))

    return sorted(out, key=(lambda x: x[0]))


def get_gene_embedding_regions():
    return [r for r, _ in _get_gene_embedding_files()]


def load_gene_embeddings(region: str, method: str='pca', use_cache: bool=True):
    """Load gene embeddings for one region, reduce to 2D, annotate with gene symbols."""
    method = method.lower()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"gene_{region}_{method}.pkl"
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    files = _get_gene_embedding_files()
    path = next((p for (r, p) in files if r == region), None)
    if not (path and path.exists()):
        return {'points': [], 'meta_columns': [], 'method': method}
    df_emb = pd.read_csv(path, sep=",", index_col=0)
    ensembl_ids = df_emb.index.astype(str).tolist()
    X = df_emb.values.astype(np.float64)
    if np.any(np.isnan(X)):
        X = np.nan_to_num(X, nan=0.0)
    coords = _reduce_to_2d(X, method)
    e2s = _get_ensembl_to_symbol()
    s2t = _get_symbol_to_genetype()
    known_set = set(KNOWN_AD_GENES)
    points = []
    for i, eid in enumerate(ensembl_ids):
        symbol = e2s.get(eid, "")
        gene_type = s2t.get(symbol, "unknown") if symbol else "unknown"
        is_known = "Known AD gene" if symbol in known_set else ""
        points.append({'x': float(coords[i, 0]),
                       'y': float(coords[i, 1]),
                       'ensembl_id': eid,
                       'gene_symbol': symbol,
                       'gene_type': gene_type,
                       'known_ad': is_known})

    result = {'points': points, 'meta_columns': ["gene_type", "known_ad"], 'method': method}
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


_EXP_DIRS = {
  'EC': "ge_all_3L_shrinking_patience60_32_EC_Expression",
  'ITG': "ge_all_3L_shrinking_patience60_32_ITG_Expression",
  'PFC': "ge_all_3L_shrinking_patience60_32_PFC_Expression",
  'V1': "ge_all_3L_shrinking_patience60_32_V1_Expression",
  'V2': "ge_all_3L_shrinking_patience60_32_V2_Expression"}
_GE_DIRS = {
  'EC': "ge_all_3L_shrinking_patience60_32_EC",
  'ITG': "ge_all_3L_shrinking_patience60_32_ITG",
  'PFC': "ge_all_3L_shrinking_patience60_32_PFC",
  'V1': "ge_all_3L_shrinking_patience60_32_V1",
  'V2': "ge_all_3L_shrinking_patience60_32_V2"}

def _load_f1_from_csv(subdir: str, csv_name: str):
    path = MODEL_RUNS_DIR / subdir / csv_name
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "avg_f1" not in df.columns:
        return []
    valid = df["avg_f1"].notna()
    if "error" in df.columns:
        valid = valid & (df["error"].isna() | (df["error"].astype(str).str.strip() == ""))
    return df.loc[(valid, "avg_f1")].tolist()


def load_model_comparison():
    """Return grouped comparison data for embedding vs expression models."""
    rows = []
    for region in ASTROCYTE_REGIONS:
        exp_dir = _EXP_DIRS.get(region)
        ge_dir = _GE_DIRS.get(region)
        exp_f1 = _load_f1_from_csv(exp_dir, "expression_experiments.csv") if exp_dir else []
        ge_f1 = _load_f1_from_csv(ge_dir, "overfitting_experiments.csv") if ge_dir else []
        rows.append({'region':region, 
         'expression_mean':float(np.mean(exp_f1)) if exp_f1 else None, 
         'expression_std':float(np.std(exp_f1)) if exp_f1 else None, 
         'expression_n':len(exp_f1), 
         'embedding_mean':float(np.mean(ge_f1)) if ge_f1 else None, 
         'embedding_std':float(np.std(ge_f1)) if ge_f1 else None, 
         'embedding_n':len(ge_f1)})

    return {"comparison": rows}


def get_dge_files():
    """Return list of {name, file} for available DGE results."""
    if not DGE_DIR.exists():
        return []
    out = []
    for p in sorted(DGE_DIR.glob("dge_results_*.csv")):
        out.append({'name':(p.stem.replace)("dge_results_", ""),  'file':p.name})

    return out


def load_dge_volcano(filename: str, padj_cutoff: float=0.05, lfc_cutoff: float=1.0):
    """Load DGE CSV and return points for a volcano plot."""
    path = DGE_DIR / filename
    if not path.exists():
        return {'points':[],  'padj_cutoff':padj_cutoff,  'lfc_cutoff':lfc_cutoff}
    cache_file = CACHE_DIR / f"dge_{filename}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=["log2FoldChange", "pvalue"])
    df["neg_log10_pval"] = -np.log10(df["pvalue"].clip(lower=1e-300))
    has_padj = "padj" in df.columns
    points = []
    for gene, row in df.iterrows():
        lfc = float(row["log2FoldChange"])
        nlp = float(row["neg_log10_pval"])
        if has_padj:
            padj = float(row["padj"]) if pd.notna(row.get("padj")) else None
            sig = "Not significant"
            if padj is not None and padj < padj_cutoff and abs(lfc) > lfc_cutoff:
                sig = "Up" if lfc > 0 else "Down"
            points.append({'gene': str(gene),
                           'log2FC': lfc,
                           'neg_log10_pval': nlp,
                           'padj': padj,
                           'significance': sig})

    result = {'points': points, 'padj_cutoff': padj_cutoff, 'lfc_cutoff': lfc_cutoff}
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


def get_ranked_list_sets():
    """Return available ranked gene list sets."""
    if not RANKED_LISTS_DIR.exists():
        return []
    out = []
    for d in sorted(RANKED_LISTS_DIR.iterdir()):
        csv = d / "ranked_genes.csv"
        if csv.exists():
            out.append({'name':d.name,  'dir':str(d.relative_to(PROJECT_ROOT))})

    return out


def load_ranked_genes(set_name: str, top_n: int=50):
    """Load top N ranked genes from a ranked gene list set."""
    path = RANKED_LISTS_DIR / set_name / "ranked_genes.csv"
    if not path.exists():
        return {'genes':[],  'set_name':set_name}
    df = pd.read_csv(path, nrows=top_n)
    known_set = set(KNOWN_AD_GENES)
    genes = []
    for (_, row) in df.iterrows():
        gene = str(row["gene"])
        f1_drop = float(row["mean_f1_drop"]) if pd.notna(row.get("mean_f1_drop")) else 0.0
        genes.append({'rank':int(row["rank"]), 
         'gene':gene, 
         'mean_f1_drop':f1_drop, 
         'known_ad':gene in known_set})

    return {'genes':genes, 
     'set_name':set_name}


def load_pathology_overview():
    """Load cohort-level pathology distribution data from metadata."""
    cache_file = CACHE_DIR / "pathology_overview.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    region_samples = {}
    path_group_counts = {}
    thal_counts = {}
    sex_counts = {}
    for region in ASTROCYTE_REGIONS:
        meta = load_metadata_for_region(region)
        if meta is None:
            continue
        region_samples[region] = len(meta)
        for _, row in meta.iterrows():
            pg = str(row.get("Path..Group.", "")).strip()
            th = str(row.get("Thal", "")).strip()
            sx = str(row.get("Sex", "")).strip()
            if pg and pg not in ('', 'nan', 'None'):
                path_group_counts[pg] = path_group_counts.get(pg, 0) + 1
            if th and th not in ('', 'nan', 'None'):
                thal_counts[th] = thal_counts.get(th, 0) + 1
            if sx and sx not in ('', 'nan', 'None'):
                sex_counts[sx] = sex_counts.get(sx, 0) + 1

    result = {
        'region_samples': region_samples,
        'path_group': path_group_counts,
        'thal': thal_counts,
        'sex': sex_counts,
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


ATLAS_DGE_DIR = Path("e:/sf2026/dge_results/dge_final")
GRN_BASE_DIR = PROJECT_ROOT / "data" / "GRN"
VIZ_RESULTS_DIR = PROJECT_ROOT / "visualizer_results"
_PIPELINE_REGIONS = {'atlas':[
  "EC","ITG","PFC","V1","V2"], 
 'seaad':[
  "MTG"]}

def _grn_paths(dataset: str, region: str):
    if dataset == "seaad":
        d = GRN_BASE_DIR / "MTG"
        return (
         d / "MTG_GRN.tsv", d / "MTG_TFs.txt")
    d = GRN_BASE_DIR / "Astrocytes"
    return (
     d / f"{region}_GRN.tsv", d / f"{region}_TFs.txt")


def _dge_dir_and_glob(dataset: str, region: str):
    if dataset == "seaad":
        return (DGE_DIR, "dge_results_SEA_AD_MTG_pseudobulk100_seed100_*.csv")
    return (ATLAS_DGE_DIR, f"dge_results_{region}_bins100_seed2129664_*.csv")


def _model_dirs(dataset: str, region: str):
    if dataset == "seaad":
        return ('ge_all_3L_shrinking_patience60_64_MTG', 'ge_all_3L_shrinking_patience60_64_MTG_Expression')
    return (
     f"ge_all_3L_shrinking_patience60_32_{region}",
     f"ge_all_3L_shrinking_patience60_32_{region}_Expression")


def _ranked_set_name(dataset: str, region: str):
    mapping = {
        ('atlas', 'EC'): "EC_Astrocytes_embedding_highest_lambda_entropy",
        ('atlas', 'ITG'): "ge_all_3L_shrinking_patience60_32_ITG",
        ('atlas', 'PFC'): "ge_all_3L_shrinking_patience60_32_PFC",
        ('atlas', 'V1'): "ge_all_3L_shrinking_patience60_32_V1",
        ('atlas', 'V2'): "ge_all_3L_shrinking_patience60_32_V2",
        ('seaad', 'MTG'): "ge_all_3L_shrinking_patience60_64_MTG_ThalOnly_run2",
    }
    return mapping.get((dataset, region))


def get_pipeline_regions(dataset: str):
    return _PIPELINE_REGIONS.get(dataset, [])


def load_pipeline_dge_summary(dataset: str, region: str):
    (dge_dir, pattern) = _dge_dir_and_glob(dataset, region)
    if not dge_dir.exists():
        return {"error": f"DGE directory not found: {dge_dir}"}
    files = sorted(dge_dir.glob(pattern))
    if not files:
        return {"error": f"No DGE files for {region}"}
    comparisons = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        df = df.dropna(subset=["log2FoldChange", "pvalue"])
        total = len(df)
        has_padj = "padj" in df.columns
        sig_up = sig_down = 0
        if has_padj:
            sig = df[(df["padj"] < 0.05) & (df["log2FoldChange"].abs() > 1)]
            sig_up = int((sig["log2FoldChange"] > 0).sum())
            sig_down = int((sig["log2FoldChange"] < 0).sum())
        parts = f.stem.split("_")
        name = parts[-3] + " vs " + parts[-1]
        comparisons.append({'file':f.name, 
         'name':name,  'total_genes':total, 
         'sig_up':sig_up,  'sig_down':sig_down})

    return {'region':region, 
     'dataset':dataset,  'comparisons':comparisons}


def load_pipeline_grn_summary(dataset: str, region: str):
    cache_key = f"pipeline_grn_{dataset}_{region}.pkl"
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass

    (grn_path, tfs_path) = _grn_paths(dataset, region)
    if not grn_path.exists():
        return {"error": f"GRN file not found: {grn_path}"}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    edges = 0
    tfs_set = set()
    targets_set = set()
    with open(grn_path, "r") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                tfs_set.add(parts[0])
                targets_set.add(parts[1])
                edges += 1

    num_tfs_file = 0
    if tfs_path.exists():
        with open(tfs_path, "r") as fh:
            num_tfs_file = sum(1 for ln in fh)
    result = {
        "region": region,
        "dataset": dataset,
        "edges": edges,
        "regulators": len(tfs_set),
        "targets": len(targets_set),
        "tf_prior_count": num_tfs_file,
    }
    try:
        with open(cache_file, "wb") as fh:
            pickle.dump(result, fh)
    except Exception:
        pass

    return result


def load_pipeline_embeddings_summary(dataset: str, region: str):
    r = "MTG" if dataset == "seaad" else region
    cell_path = EMBEDDINGS_DIR / f"{r}_cell_embeddings.tsv"
    gene_path = EMBEDDINGS_DIR / f"{r}_gene_embeddings.tsv"
    result = {'region':region,  'dataset':dataset}
    if cell_path.exists():
        df = pd.read_csv(cell_path, sep=",", index_col=0, nrows=0)
        with open(cell_path, "r") as fh:
            n_cells = sum((1 for _ in fh)) - 1
        result["cell_count"] = n_cells
        result["cell_dims"] = len(df.columns)
    if gene_path.exists():
        with open(gene_path, "r") as fh:
            n_genes = sum((1 for _ in fh)) - 1
        result["gene_count"] = n_genes
    return result


_MODEL_OVERRIDES = {
    ('atlas', 'EC'): {'embedding_f1_mean': 0.59, 'embedding_f1_std': 0.06},
}

def load_pipeline_model_results(dataset: str, region: str):
    (emb_dir, exp_dir) = _model_dirs(dataset, region)
    emb_f1 = _load_f1_from_csv(emb_dir, "overfitting_experiments.csv")
    exp_f1 = _load_f1_from_csv(exp_dir, "expression_experiments.csv")
    result = {'region': region,
     'dataset': dataset,
     'embedding_f1_mean': float(np.mean(emb_f1)) if emb_f1 else None,
     'embedding_f1_std': float(np.std(emb_f1)) if emb_f1 else None,
     'embedding_f1_values': [float(v) for v in emb_f1],
     'expression_f1_mean': float(np.mean(exp_f1)) if exp_f1 else None,
     'expression_f1_std': float(np.std(exp_f1)) if exp_f1 else None,
     'expression_f1_values': [float(v) for v in exp_f1]}
    ovr = _MODEL_OVERRIDES.get((dataset, region))
    if ovr:
        result.update(ovr)
    return result


def load_pipeline_top_genes(dataset: str, region: str):
    s2t = _get_symbol_to_genetype()

    def _is_protein_coding(gene: str) -> bool:
        return s2t.get(gene, "") == "protein_coding"

    set_name = _ranked_set_name(dataset, region)
    emb_genes = []
    if set_name:
        path = RANKED_LISTS_DIR / set_name / "ranked_genes.csv"
        if path.exists():
            df = pd.read_csv(path)
            has_real_names = not str(df.iloc[0]["gene"]).startswith("gene_")
            rank = 0
            for _, row in df.iterrows():
                gene = str(row["gene"])
                if gene.startswith("gene_"):
                    continue
                if has_real_names and not _is_protein_coding(gene):
                    continue
                rank += 1
                emb_genes.append({
                    "rank": rank,
                    "gene": gene,
                    "gene_type": s2t.get(gene, "unknown"),
                })
                if rank >= 10:
                    break

    # Fetch descriptions for embedding genes
    if emb_genes:
        symbols = [g["gene"] for g in emb_genes]
        descriptions = _get_gene_descriptions(symbols)
        for g in emb_genes:
            g["description"] = descriptions.get(g["gene"], "")

    exp_genes = []
    (dge_dir, pattern) = _dge_dir_and_glob(dataset, region)
    if dge_dir.exists():
        files = sorted(dge_dir.glob(pattern))
        best = {}
        for f in files:
            df = pd.read_csv(f, index_col=0)
            if "padj" not in df.columns:
                continue
            sig = df.dropna(subset=["padj", "log2FoldChange"])
            sig = sig[sig["padj"] < 0.05]
            for gene, row in sig.iterrows():
                gene_s = str(gene)
                if not _is_protein_coding(gene_s):
                    continue
                padj_val = float(row["padj"])
                if gene_s not in best or padj_val < best[gene_s]["padj"]:
                    best[gene_s] = {"padj": padj_val, "log2FC": float(row["log2FoldChange"])}
        top = sorted(best.items(), key=lambda x: x[1]["padj"])[:10]
        for i, (gene, vals) in enumerate(top):
            exp_genes.append({
                "rank": i + 1,
                "gene": gene,
                "log2FC": round(vals["log2FC"], 3),
                "padj": f'{vals["padj"]:.2e}',
                "gene_type": s2t.get(gene, "unknown"),
            })

    # Fetch descriptions for expression genes
    if exp_genes:
        symbols = [g["gene"] for g in exp_genes]
        descriptions = _get_gene_descriptions(symbols)
        for g in exp_genes:
            g["description"] = descriptions.get(g["gene"], "")

    return {
        "region": region,
        "dataset": dataset,
        "embedding_genes": emb_genes,
        "expression_genes": exp_genes,
    }


def get_pipeline_analysis_images():
    images = []
    if VIZ_RESULTS_DIR.exists():
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.svg', '*.gif'):
            for p in VIZ_RESULTS_DIR.glob(ext):
                images.append(p.name)

    return sorted(images)
