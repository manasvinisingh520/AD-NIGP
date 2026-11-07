"""
Export the embedding_viz Flask demo as a static site (for GitHub Pages).

It calls every API URL the page can request, saves each response as a JSON
file, copies the static assets, and writes an index.html with a small shim
that redirects `fetch('/api/...')` to those JSON files.

Run from project root:  python -m embedding_viz.export_static [--out site]
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlencode

from embedding_viz.app import app
from embedding_viz.load_data import SUPPORTED_METHODS, VIZ_RESULTS_DIR

STATIC_DIR = Path(__file__).parent / "static"
TOP_N_OPTIONS = (25, 50, 100, 200)       # must match <select id="rank-topn">
PIPELINE_DATASETS = ("atlas", "seaad")   # must match data-ds buttons


def static_name(path, params):
    """Map an API path + query params to a relative JSON file path.
    Must stay identical to apiFile() in the JS shim below."""
    name = path.lstrip("/")
    if params:
        q = "__".join(f"{k}-{params[k]}" for k in sorted(params))
        name += "__" + re.sub(r"[^A-Za-z0-9._-]", "_", q)
    return name + ".json"


SHIM = r"""
<script>
// Static-site shim: serve /api/... requests from pre-exported JSON files.
(function () {
  function apiFile(url) {
    const u = new URL(url, 'http://x');
    const keys = [...u.searchParams.keys()].sort();
    let name = u.pathname.replace(/^\//, '');
    if (keys.length) {
      const q = keys.map(k => k + '-' + u.searchParams.get(k)).join('__');
      name += '__' + q.replace(/[^A-Za-z0-9._-]/g, '_');
    }
    return name + '.json';
  }
  const realFetch = window.fetch.bind(window);
  window.fetch = function (url, opts) {
    if (typeof url === 'string' && url.startsWith('/api/')) {
      return realFetch(apiFile(url), opts).then(r => r.ok ? r :
        new Response(JSON.stringify({error: 'Not available in the static demo'}),
                     {status: 404, headers: {'Content-Type': 'application/json'}}));
    }
    return realFetch(url, opts);
  };
})();
</script>
"""


class Exporter:
    def __init__(self, out):
        self.out = out
        self.client = app.test_client()
        self.failed = []

    def get(self, path, **params):
        """Fetch one API URL, save it, and return the parsed JSON."""
        url = path + ("?" + urlencode(params) if params else "")
        resp = self.client.get(url)
        data = resp.get_json(silent=True)
        if resp.status_code != 200 or data is None:
            self.failed.append((url, resp.status_code))
            print(f"  FAILED {resp.status_code}: {url}")
            return None
        dest = self.out / static_name(path, params)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        print(f"  {dest.stat().st_size / 1e6:8.2f} MB  {url}")
        return data

    def run(self):
        print("Cell embeddings")
        for region in self.get("/api/regions")["regions"]:
            for m in SUPPORTED_METHODS:
                self.get("/api/embeddings", region=region, method=m)

        print("Gene embeddings")
        for region in self.get("/api/gene_regions")["regions"]:
            for m in SUPPORTED_METHODS:
                self.get("/api/gene_embeddings", region=region, method=m)

        print("Model comparison / pathology")
        self.get("/api/model_comparison")
        self.get("/api/pathology_overview")

        print("DGE volcano")
        # The page sends each entry's "file" field (see initVolcano in index.html)
        for f in self.get("/api/dge_files")["files"]:
            self.get("/api/dge_volcano", file=f["file"])

        print("Gene rankings")
        # The page sends each entry's "name" field (see populateSelect for rankSet)
        for s in self.get("/api/ranked_sets")["sets"]:
            for n in TOP_N_OPTIONS:
                self.get("/api/ranked_genes", set=s["name"], top=str(n))

        print("Interactive pipeline")
        for ds in PIPELINE_DATASETS:
            regions = self.get("/api/pipeline/regions", dataset=ds)["regions"]
            for region in regions:
                code = region["code"] if isinstance(region, dict) else region
                for step in ("dge", "grn", "embeddings", "model", "top_genes"):
                    self.get(f"/api/pipeline/{step}", dataset=ds, region=code)
        self.get("/api/pipeline/analysis_images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site", help="output folder (default: site)")
    out = Path(ap.parse_args().out)

    # 1. Static assets (images etc.)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC_DIR, out)
    if VIZ_RESULTS_DIR.exists():
        shutil.copytree(VIZ_RESULTS_DIR, out / "visualizer_results")

    # 2. All API responses as JSON
    ex = Exporter(out)
    ex.run()

    # 3. index.html: add shim, make absolute paths relative (Pages serves under /<repo>/)
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("/visualizer_results/", "visualizer_results/")
    html = html.replace("<head>", "<head>" + SHIM, 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("")  # don't let GitHub Pages run Jekyll

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\nDone: {out}  ({total / 1e6:.1f} MB total, {len(ex.failed)} failed)")
    for url, code in ex.failed:
        print(f"  failed {code}: {url}")


if __name__ == "__main__":
    main()
