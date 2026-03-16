"""
Flask app for interactive GREmLN embedding visualization demo.
Run from project root: python -m embedding_viz.app
"""

import os
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, request

from embedding_viz.load_data import (
    get_available_regions,
    load_region_embeddings_and_metadata,
    load_all_regions_embeddings,
    ALL_REGIONS_KEY,
    SUPPORTED_METHODS,
    get_gene_embedding_regions,
    load_gene_embeddings,
    load_model_comparison,
    get_dge_files,
    load_dge_volcano,
    get_ranked_list_sets,
    load_ranked_genes,
    load_pathology_overview,
    get_pipeline_regions,
    load_pipeline_dge_summary,
    load_pipeline_grn_summary,
    load_pipeline_embeddings_summary,
    load_pipeline_model_results,
    load_pipeline_top_genes,
    get_pipeline_analysis_images,
    VIZ_RESULTS_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---- Cell Embeddings ----

@app.route("/api/regions")
def api_regions():
    regions = get_available_regions()
    return jsonify({"regions": [ALL_REGIONS_KEY] + regions})


@app.route("/api/embeddings")
def api_embeddings():
    region = request.args.get("region", "").strip()
    if not region:
        return jsonify({"error": "Missing region"}), 400
    method = request.args.get("method", "pca").lower()
    if method not in SUPPORTED_METHODS:
        return jsonify({"error": f"Unknown method '{method}'"}), 400
    use_cache = request.args.get("cache", "true").lower() == "true"
    try:
        if region == ALL_REGIONS_KEY:
            data = load_all_regions_embeddings(method=method, use_cache=use_cache)
        else:
            data = load_region_embeddings_and_metadata(region, method=method, use_cache=use_cache)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Gene Embeddings ----

@app.route("/api/gene_regions")
def api_gene_regions():
    return jsonify({"regions": get_gene_embedding_regions()})


@app.route("/api/gene_embeddings")
def api_gene_embeddings():
    region = request.args.get("region", "").strip()
    if not region:
        return jsonify({"error": "Missing region"}), 400
    method = request.args.get("method", "pca").lower()
    if method not in SUPPORTED_METHODS:
        return jsonify({"error": f"Unknown method '{method}'"}), 400
    use_cache = request.args.get("cache", "true").lower() == "true"
    try:
        data = load_gene_embeddings(region, method=method, use_cache=use_cache)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Model Comparison ----

@app.route("/api/model_comparison")
def api_model_comparison():
    try:
        return jsonify(load_model_comparison())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- DGE Volcano ----

@app.route("/api/dge_files")
def api_dge_files():
    return jsonify({"files": get_dge_files()})


@app.route("/api/dge_volcano")
def api_dge_volcano():
    filename = request.args.get("file", "").strip()
    if not filename:
        return jsonify({"error": "Missing file"}), 400
    try:
        data = load_dge_volcano(filename)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Gene Rankings ----

@app.route("/api/ranked_sets")
def api_ranked_sets():
    return jsonify({"sets": get_ranked_list_sets()})


@app.route("/api/ranked_genes")
def api_ranked_genes():
    set_name = request.args.get("set", "").strip()
    if not set_name:
        return jsonify({"error": "Missing set"}), 400
    top_n = int(request.args.get("top", "50"))
    try:
        return jsonify(load_ranked_genes(set_name, top_n=top_n))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Pathology Overview ----

@app.route("/api/pathology_overview")
def api_pathology_overview():
    try:
        return jsonify(load_pathology_overview())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Interactive Pipeline ----

@app.route("/api/pipeline/regions")
def api_pipeline_regions():
    dataset = request.args.get("dataset", "").strip().lower()
    return jsonify({"regions": get_pipeline_regions(dataset)})


@app.route("/api/pipeline/dge")
def api_pipeline_dge():
    dataset = request.args.get("dataset", "").strip().lower()
    region = request.args.get("region", "").strip()
    try:
        return jsonify(load_pipeline_dge_summary(dataset, region))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/grn")
def api_pipeline_grn():
    dataset = request.args.get("dataset", "").strip().lower()
    region = request.args.get("region", "").strip()
    try:
        return jsonify(load_pipeline_grn_summary(dataset, region))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/embeddings")
def api_pipeline_embeddings():
    dataset = request.args.get("dataset", "").strip().lower()
    region = request.args.get("region", "").strip()
    try:
        return jsonify(load_pipeline_embeddings_summary(dataset, region))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/model")
def api_pipeline_model():
    dataset = request.args.get("dataset", "").strip().lower()
    region = request.args.get("region", "").strip()
    try:
        return jsonify(load_pipeline_model_results(dataset, region))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/top_genes")
def api_pipeline_top_genes():
    dataset = request.args.get("dataset", "").strip().lower()
    region = request.args.get("region", "").strip()
    try:
        return jsonify(load_pipeline_top_genes(dataset, region))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/analysis_images")
def api_pipeline_images():
    return jsonify({"images": get_pipeline_analysis_images()})


@app.route("/visualizer_results/<path:filename>")
def serve_viz_results(filename):
    return send_from_directory(str(VIZ_RESULTS_DIR), filename)


def main():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
