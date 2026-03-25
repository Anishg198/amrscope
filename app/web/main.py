"""
AMRScope — FastAPI web application
Run:  uvicorn app.web.main:app --reload --port 8000
"""
import sys, pickle, json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

ROOT       = Path(__file__).resolve().parents[2]
WEB_DIR    = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GRAPH_PATH  = ROOT / "data/processed/biomolamr_graph.pkl"
SPLITS_PATH = ROOT / "data/processed/extended_splits.pkl"
MODELS_DIR  = ROOT / "results/biomolamr/models"
EVAL_PATH   = ROOT / "results/biomolamr/eval_summary.json"
FIGS_DIR    = ROOT / "results/biomolamr/figures"
DEVICE      = torch.device("cpu")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AMRScope")
app.mount("/static",  StaticFiles(directory=str(WEB_DIR / "static")),   name="static")
app.mount("/figures", StaticFiles(directory=str(FIGS_DIR)),              name="figures")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

# ── Startup: load everything once ─────────────────────────────────────────────
_graph_obj = None
_splits    = None
_eval      = None
_models    = {}   # name → model
_emb_cache = {}   # name → (gene_emb, drug_emb) pre-computed for compare
_gene_list = []   # [{idx, name, aro, description}]
_drug_list = []   # [{idx, name, is_zs}]
_zs_set    = set()
_pos_pairs = set()

@app.on_event("startup")
def startup():
    global _graph_obj, _splits, _eval, _gene_list, _drug_list, _zs_set, _pos_pairs, _emb_cache

    print("Loading graph…")
    with open(GRAPH_PATH, "rb") as f:
        _graph_obj = pickle.load(f)
    print("Loading splits…")
    with open(SPLITS_PATH, "rb") as f:
        _splits = pickle.load(f)
    print("Loading eval…")
    with open(EVAL_PATH) as f:
        _eval = json.load(f)

    dc2i      = _graph_obj["dc2i"]
    gene2idx  = _graph_obj["gene2idx"]
    gene_meta = _graph_obj["gene_metadata"]
    idx2aro   = {v: k for k, v in gene2idx.items()}

    _zs_set   = set(_splits["meta"]["zs_drug_class_names"])
    _pos_pairs = set(map(tuple, _splits["meta"]["pos_pairs"]))

    # Build gene list
    for aro, idx in sorted(gene2idx.items(), key=lambda x: x[1]):
        gm   = gene_meta.get(aro, {})
        name = gm.get("name", aro)       if isinstance(gm, dict) else aro
        desc = gm.get("description", "") if isinstance(gm, dict) else ""
        _gene_list.append({"idx": idx, "name": name, "aro": aro, "description": desc})

    # Build drug list
    for dc_name, dc_idx in sorted(dc2i.items(), key=lambda x: x[1]):
        _drug_list.append({"idx": dc_idx, "name": dc_name, "is_zs": dc_name in _zs_set})

    # Load models (seed=42)
    print("Loading models…")
    _load_all_models()
    # Pre-compute embeddings for fast compare
    print("Pre-computing embeddings…")
    _precompute_embeddings()
    print("Ready.")


def _load_all_models():
    from src.models.baselines  import FeatureMLPBaseline, RGCNBaseline
    from src.models.biomolamr  import BioMolAMR as AMRScope

    g = _graph_obj["hetero_data"]
    g_dim, d_dim, m_dim = (g["gene"].x.shape[1], g["drug_class"].x.shape[1],
                           g["mechanism"].x.shape[1])

    specs = {
        "feature_mlp": MODELS_DIR / "feature_mlp_seed42.pt",
        "rgcn_bio":    MODELS_DIR / "rgcn_bio_seed42.pt",
        "amrscope":   MODELS_DIR / "biomolamr_seed42.pt",
    }
    for name, path in specs.items():
        if not path.exists():
            print(f"  Skipping {name} — checkpoint not found")
            continue
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        hp = ck["hparams"]
        if name == "feature_mlp":
            m = FeatureMLPBaseline(g_dim, d_dim, hp["hidden_dim"], hp["dropout"])
        elif name == "rgcn_bio":
            m = RGCNBaseline(g_dim, d_dim, m_dim, hp["hidden_dim"], hp["out_dim"], hp["dropout"])
        else:
            m = AMRScope(g_dim, d_dim, m_dim, hp["hidden_dim"], hp["out_dim"],
                          hp["num_heads"], hp["num_gat_layers"], hp["dropout"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        _models[name] = m
        print(f"  Loaded {name}")


def _precompute_embeddings():
    """Run each GNN model once over the full graph to cache all embeddings."""
    g = _graph_obj["hetero_data"]
    n_g  = len(_gene_list)
    n_dc = len(_drug_list)
    all_g = torch.arange(n_g,  dtype=torch.long)
    all_d = torch.arange(n_dc, dtype=torch.long)
    for mn, m in _models.items():
        with torch.no_grad():
            if mn == "feature_mlp":
                # MLP is already fast — store raw feature matrices
                _emb_cache[mn] = (g["gene"].x, g["drug_class"].x)
            else:
                # For GNNs, run a dummy forward to warm up; scores computed per-request
                # but cache the full score matrix (n_g x n_dc) at startup
                try:
                    scores = np.zeros((n_g, n_dc), dtype=np.float32)
                    for di in range(n_dc):
                        s = m(g, all_g, torch.full((n_g,), di, dtype=torch.long))
                        scores[:, di] = s.cpu().numpy()
                    _emb_cache[mn] = scores
                    print(f"  Cached {mn} score matrix")
                except Exception as e:
                    print(f"  Could not cache {mn}: {e}")
                    _emb_cache[mn] = None


def _score(model_name, gene_indices, drug_indices):
    """Score a batch of (gene, drug) pairs. Returns numpy array."""
    m  = _models[model_name]
    g  = _graph_obj["hetero_data"]
    gt = torch.tensor(gene_indices, dtype=torch.long)
    dt = torch.tensor(drug_indices,  dtype=torch.long)
    with torch.no_grad():
        if model_name == "feature_mlp":
            s = m(g["gene"].x, g["drug_class"].x, gt, dt)
        elif model_name == "rgcn_bio":
            s = m(g, gt, dt)
        else:
            s = m(g, gt, dt)
    return s.cpu().numpy()


# ── HTML pages ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "n_genes": len(_gene_list),
        "n_drugs": len(_drug_list),
        "n_zs":    len(_zs_set),
    })

@app.get("/results", response_class=HTMLResponse)
def results_page(request: Request):
    return templates.TemplateResponse("results.html", {
        "request": request,
        "eval":    _eval,
    })

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {
        "request":    request,
        "gene_list":  _gene_list[:200],   # first 200 for <select>; rest via search
        "drug_list":  _drug_list,
        "models":     list(_models.keys()),
    })

@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


# ── JSON API ───────────────────────────────────────────────────────────────────

@app.get("/api/genes")
def api_genes(q: str = ""):
    """Search genes by name or ARO."""
    q = q.lower().strip()
    if not q:
        return _gene_list[:500]
    return [g for g in _gene_list if q in g["name"].lower() or q in g["aro"]][:100]

@app.get("/api/drugs")
def api_drugs(q: str = ""):
    """Return drug list, optionally filtered by name search."""
    if not q:
        return _drug_list
    q = q.lower().strip()
    return [d for d in _drug_list if q in d["name"].lower()][:50]

@app.get("/api/results")
def api_results():
    return _eval

@app.get("/api/summary_stats")
def api_summary():
    return {
        "n_genes":     len(_gene_list),
        "n_drugs":     len(_drug_list),
        "n_zs":        len(_zs_set),
        "models":      list(_models.keys()),
        "random_all":  round(1/46, 4),
        "random_win":  round(1/12, 4),
    }


class GeneDrugRequest(BaseModel):
    gene_idx:   int
    model_name: str = "feature_mlp"

@app.post("/api/predict/gene-drug")
def predict_gene_drug(req: GeneDrugRequest):
    if req.model_name not in _models:
        raise HTTPException(404, f"Model '{req.model_name}' not loaded")
    n_dc   = len(_drug_list)
    scores = _score(req.model_name,
                    [req.gene_idx] * n_dc,
                    list(range(n_dc)))
    order  = np.argsort(scores)[::-1]
    true_drugs = {d for (g, d) in _pos_pairs if g == req.gene_idx}
    out = []
    for rank, dc_idx in enumerate(order, 1):
        dc_idx = int(dc_idx)
        dc = _drug_list[dc_idx]
        out.append({
            "rank":    rank,
            "dc_idx":  dc_idx,
            "name":    dc["name"],
            "score":   float(scores[dc_idx]),
            "is_zs":   dc["is_zs"],
            "is_tp":   dc_idx in true_drugs,
        })
    return out


class DrugGeneRequest(BaseModel):
    drug_idx:   int
    model_name: str = "feature_mlp"
    top_k:      int = 30

@app.post("/api/predict/drug-gene")
def predict_drug_gene(req: DrugGeneRequest):
    if req.model_name not in _models:
        raise HTTPException(404, f"Model '{req.model_name}' not loaded")
    n_g    = len(_gene_list)
    scores = _score(req.model_name,
                    list(range(n_g)),
                    [req.drug_idx] * n_g)
    order  = np.argsort(scores)[::-1][:req.top_k]
    true_genes = {g for (g, d) in _pos_pairs if d == req.drug_idx}
    out = []
    for rank, gidx in enumerate(order, 1):
        gidx = int(gidx)
        g = _gene_list[gidx]
        out.append({
            "rank":        rank,
            "gene_idx":    gidx,
            "name":        g["name"],
            "aro":         g["aro"],
            "score":       float(scores[gidx]),
            "is_tp":       gidx in true_genes,
            "description": g["description"][:80],
        })
    n_tp = len(true_genes)
    n_found = sum(1 for r in out if r["is_tp"])
    return {"results": out, "n_true_positives": n_tp,
            "precision_at_k": round(n_found / req.top_k, 3)}


class SmilesRequest(BaseModel):
    smiles: str
    top_k:  int = 30

@app.post("/api/predict/smiles")
def predict_smiles(req: SmilesRequest):
    # Compute fingerprint
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit.Chem.AtomPairs import Torsions
        mol = Chem.MolFromSmiles(req.smiles.strip())
        if mol is None:
            raise HTTPException(400, "Invalid SMILES")
        morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        maccs  = list(MACCSkeys.GenMACCSKeys(mol))
        # Use GetNonzeroElements() to avoid iterating the full sparse hash space
        topo_sparse = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
        topo_nz     = topo_sparse.GetNonzeroElements()
        topo_b      = [1 if v > 0 else 0 for v in topo_nz.values()]
        fp     = (morgan + maccs + topo_b)[:3245]
        fp    += [0] * max(0, 3245 - len(fp))
        fp_t   = torch.tensor(fp, dtype=torch.float32)
    except ImportError:
        raise HTTPException(500, "RDKit not installed")

    # Tanimoto to nearest training drug
    g_data  = _graph_obj["hetero_data"]
    db_fp   = g_data["drug_class"].x.numpy().astype(float)
    fp_np   = np.array(fp, dtype=float)
    inter   = (db_fp * fp_np).sum(axis=1)
    union   = fp_np.sum() + db_fp.sum(axis=1) - inter + 1e-9
    sims    = inter / union
    best_dc = int(np.argmax(sims))
    best_sim = float(sims[best_dc])
    nearest_name = _drug_list[best_dc]["name"]

    # Score with Feature-MLP using novel fingerprint
    m       = _models.get("feature_mlp")
    if m is None:
        raise HTTPException(500, "Feature-MLP not loaded")

    novel_drug_x = torch.cat([g_data["drug_class"].x, fp_t.unsqueeze(0)], dim=0)
    n_genes      = g_data["gene"].x.shape[0]
    novel_dc_idx = g_data["drug_class"].x.shape[0]  # 46
    gene_t       = torch.arange(n_genes, dtype=torch.long)
    drug_t       = torch.full((n_genes,), novel_dc_idx, dtype=torch.long)

    with torch.no_grad():
        scores = m(g_data["gene"].x, novel_drug_x, gene_t, drug_t).numpy()

    order   = np.argsort(scores)[::-1][:req.top_k]
    out = []
    for rank, gidx in enumerate(order, 1):
        gidx = int(gidx)
        g = _gene_list[gidx]
        out.append({
            "rank":        rank,
            "gene_idx":    gidx,
            "name":        g["name"],
            "aro":         g["aro"],
            "score":       float(scores[gidx]),
            "description": g["description"][:80],
        })
    return {
        "results":       out,
        "nearest_drug":  nearest_name,
        "tanimoto":      round(best_sim, 3),
        "fp_dim":        len(fp),
    }


class CompareRequest(BaseModel):
    gene_idx: int

@app.post("/api/predict/compare")
def predict_compare(req: CompareRequest):
    n_dc = len(_drug_list)
    result = {}
    for mn in _models:
        cache = _emb_cache.get(mn)
        if mn == "feature_mlp":
            # Use cached feature matrices — fast MLP
            scores = _score(mn, [req.gene_idx]*n_dc, list(range(n_dc)))
        elif isinstance(cache, np.ndarray):
            # Use pre-computed full score matrix
            scores = cache[req.gene_idx]
        else:
            # Fallback: compute on-the-fly
            scores = _score(mn, [req.gene_idx]*n_dc, list(range(n_dc)))
        order = np.argsort(scores)[::-1]
        result[mn] = [{"dc_idx": int(i), "name": _drug_list[i]["name"],
                        "score": float(scores[i]), "is_zs": _drug_list[i]["is_zs"]}
                      for i in order[:20]]
    return result
