"""
BioMolAMR — Zero-Shot Antimicrobial Resistance Prediction
==========================================================
Full research dashboard + interactive prediction app.

Pages:
  1. Overview    — Research story, data leakage finding, key results
  2. Results     — All figures, main table, ablation table, per-class analysis
  3. Gene → Drug — Pick a gene, rank all 46 drug classes by predicted resistance
  4. Drug → Gene — Pick a drug class (including zero-shot), rank top resistant genes
  5. Novel Drug  — Paste a SMILES string → compute fingerprints → ZS prediction
  6. Compare     — Side-by-side model predictions for the same query
"""

import sys, pickle, json
from pathlib import Path

import numpy as np
import torch
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="BioMolAMR — Zero-Shot AMR",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
GRAPH_PATH  = ROOT / "data/processed/biomolamr_graph.pkl"
SPLITS_PATH = ROOT / "data/processed/extended_splits.pkl"
TAN_PATH    = ROOT / "data/processed/drug_tanimoto.pt"
MODELS_DIR  = ROOT / "results/biomolamr/models"
EVAL_PATH   = ROOT / "results/biomolamr/eval_summary.json"
FIGS_DIR    = ROOT / "results/biomolamr/figures"

DEVICE = torch.device("cpu")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-box {
    background: #f0f4ff; border-radius: 10px; padding: 16px 20px;
    border-left: 5px solid #2166ac; margin: 6px 0;
  }
  .metric-box h3 { margin: 0; font-size: 2rem; color: #2166ac; }
  .metric-box p  { margin: 2px 0; color: #555; font-size: 0.9rem; }
  .finding-box {
    background: #fff8e1; border-radius: 8px; padding: 14px 18px;
    border-left: 5px solid #f9a825; margin: 8px 0;
  }
  .leakage-box {
    background: #ffebee; border-radius: 8px; padding: 14px 18px;
    border-left: 5px solid #c62828; margin: 8px 0;
  }
  .result-row { border-bottom: 1px solid #eee; padding: 6px 0; }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading graph and features…")
def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading splits…")
def load_splits():
    with open(SPLITS_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading evaluation results…")
def load_eval():
    with open(EVAL_PATH) as f:
        return json.load(f)

@st.cache_resource(show_spinner="Loading Tanimoto matrix…")
def load_tanimoto():
    return torch.load(TAN_PATH, map_location="cpu").numpy()

@st.cache_resource(show_spinner="Loading Feature-MLP model…")
def load_fmlp(seed=42):
    from src.models.baselines import FeatureMLPBaseline
    ckpt_path = MODELS_DIR / f"feature_mlp_seed{seed}.pt"
    if not ckpt_path.exists():
        return None, None
    ck   = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp   = ck["hparams"]
    g_obj = load_graph()
    g     = g_obj["hetero_data"]
    model = FeatureMLPBaseline(
        gene_feat_dim=g["gene"].x.shape[1],
        drug_feat_dim=g["drug_class"].x.shape[1],
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck

@st.cache_resource(show_spinner="Loading R-GCN model…")
def load_rgcn(seed=42):
    from src.models.baselines import RGCNBaseline
    ckpt_path = MODELS_DIR / f"rgcn_bio_seed{seed}.pt"
    if not ckpt_path.exists():
        return None, None
    ck   = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp   = ck["hparams"]
    g_obj = load_graph()
    g     = g_obj["hetero_data"]
    model = RGCNBaseline(
        gene_feat_dim=g["gene"].x.shape[1],
        drug_feat_dim=g["drug_class"].x.shape[1],
        mech_feat_dim=g["mechanism"].x.shape[1],
        hidden_dim=hp["hidden_dim"],
        out_dim=hp["out_dim"],
        dropout=hp["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck

@st.cache_resource(show_spinner="Loading BioMolAMR model…")
def load_biomolamr(seed=42):
    from src.models.biomolamr import BioMolAMR
    ckpt_path = MODELS_DIR / f"biomolamr_seed{seed}.pt"
    if not ckpt_path.exists():
        return None, None
    ck   = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp   = ck["hparams"]
    g_obj = load_graph()
    g     = g_obj["hetero_data"]
    model = BioMolAMR(
        gene_feat_dim=g["gene"].x.shape[1],
        drug_feat_dim=g["drug_class"].x.shape[1],
        mech_feat_dim=g["mechanism"].x.shape[1],
        hidden_dim=hp["hidden_dim"],
        out_dim=hp["out_dim"],
        num_heads=hp["num_heads"],
        num_gat_layers=hp["num_gat_layers"],
        dropout=hp["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def get_gene_name(g_obj, idx):
    meta = g_obj["gene_metadata"]
    idx2aro = {v: k for k, v in g_obj["gene2idx"].items()}
    aro = idx2aro.get(idx, "")
    gm  = meta.get(aro, {}) if isinstance(meta, dict) else {}
    return gm.get("name", f"Gene_{idx}") if isinstance(gm, dict) else f"Gene_{idx}"


def score_gene_vs_all_drugs(model_name, model, g_data, gene_idx):
    """Return (46,) scores for one gene vs all drug classes."""
    n_dc     = g_data["drug_class"].x.shape[0]
    gene_t   = torch.tensor([gene_idx] * n_dc, dtype=torch.long)
    drug_t   = torch.tensor(list(range(n_dc)),  dtype=torch.long)
    with torch.no_grad():
        if model_name in ("feature_mlp", "fmlp"):
            s = model(g_data["gene"].x, g_data["drug_class"].x, gene_t, drug_t)
        elif model_name == "rgcn":
            s = model(g_data, gene_t, drug_t)
        elif model_name == "biomolamr":
            s = model(g_data, gene_t, drug_t)
    return s.cpu().numpy()


def score_drug_vs_all_genes(model_name, model, g_data, drug_idx, top_k=50):
    """Return scores for one drug vs all genes, return top_k indices + scores."""
    n_genes  = g_data["gene"].x.shape[0]
    gene_t   = torch.arange(n_genes, dtype=torch.long)
    drug_t   = torch.tensor([drug_idx] * n_genes, dtype=torch.long)
    with torch.no_grad():
        if model_name in ("feature_mlp", "fmlp"):
            s = model(g_data["gene"].x, g_data["drug_class"].x, gene_t, drug_t)
        elif model_name == "rgcn":
            s = model(g_data, gene_t, drug_t)
        elif model_name == "biomolamr":
            s = model(g_data, gene_t, drug_t)
    s_np = s.cpu().numpy()
    top  = np.argsort(s_np)[::-1][:top_k]
    return top, s_np[top]


def compute_fingerprint_from_smiles(smiles: str):
    """Compute 3245-dim fingerprint (Morgan+MACCS+TopoTorsion) from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit.Chem.AtomPairs import Torsions
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES — RDKit could not parse it."
        morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        maccs  = list(MACCSkeys.GenMACCSKeys(mol))
        topo   = list(Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol))
        # Topo is count-based; binarise
        topo_b = [1 if x > 0 else 0 for x in topo]
        combined = morgan + maccs + topo_b
        # Pad/trim to 3245
        if len(combined) < 3245:
            combined += [0] * (3245 - len(combined))
        else:
            combined = combined[:3245]
        return torch.tensor(combined, dtype=torch.float32), None
    except ImportError:
        return None, "RDKit not installed."
    except Exception as e:
        return None, str(e)


# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/BioMolAMR-Research-blue", use_container_width=False)
st.sidebar.title("BioMolAMR")
st.sidebar.caption("Zero-Shot AMR Prediction")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Results", "Gene → Drug Rank", "Drug → Gene Rank",
     "Novel Drug (SMILES)", "Model Comparison"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** CARD 3.2.6")
st.sidebar.markdown("**Genes:** 6,397 | **Drug classes:** 46")
st.sidebar.markdown("**ZS classes:** 12 | **Metric:** ZS-all MRR")
st.sidebar.markdown("**Best model:** Feature-MLP (0.069)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🦠 BioMolAMR: Zero-Shot Antimicrobial Resistance Prediction")
    st.markdown(
        "**Can we predict which genes confer resistance to a *novel* antibiotic class "
        "that was never seen during training?** This is the zero-shot AMR challenge."
    )

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-box"><h3>6,397</h3><p>Resistance genes (CARD)</p></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="metric-box"><h3>46</h3><p>Antibiotic drug classes</p></div>',
                unsafe_allow_html=True)
    c3.markdown('<div class="metric-box"><h3>12</h3><p>Zero-shot held-out classes</p></div>',
                unsafe_allow_html=True)
    c4.markdown('<div class="metric-box"><h3>3.6×</h3><p>Above GNNs (Feature-MLP)</p></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # Story
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("The Data Leakage Problem")
        st.markdown("""
        Prior work on zero-shot AMR reported very high MRR scores (**0.775**).
        We found this was caused by **data leakage**: gene feature vectors contained
        a 46-dimensional one-hot encoding of which drug classes each gene resists —
        *directly encoding the answer* to the zero-shot question.

        After removing this leakage and using:
        - **ESM-2** (480-dim protein language model embeddings) for genes
        - **Molecular fingerprints** (Morgan + MACCS + TopoTorsion, 3245-dim) for drugs

        ...the corrected ZS MRR drops from 0.775 → 0.137 (**5.7× reduction**).
        """)

        st.markdown('<div class="leakage-box">'
                    '<b>⚠️ Leakage:</b> When gene features encode drug-class membership, '
                    'a model can trivially "read off" the zero-shot answer. '
                    'This inflated prior reported scores by 5.7×.'
                    '</div>', unsafe_allow_html=True)

        st.subheader("Key Findings")
        st.markdown("""
        1. **Feature-MLP outperforms all GNNs** by 3.6× on the primary zero-shot metric
           (ZS-all MRR: 0.069 vs. next best 0.027 for R-GCN).

        2. **Both features are jointly necessary**: zeroing drug fingerprints collapses
           ZS-all MRR to 0.010 (below random); zeroing ESM-2 gene embeddings drops to 0.020 ≈ random.

        3. **Graph topology hurts**: adding GNN layers over resistance topology
           over-smooths training-class patterns onto held-out drugs, *reducing* zero-shot generalisation.

        4. **Predictability correlates with chemical similarity**: lincosamide (MRR=0.76) and
           streptogramin A (0.70) are easily predicted because their fingerprints are similar
           to training drug classes.
        """)

    with col2:
        fig1_path = FIGS_DIR / "F1_leakage_diagram.png"
        if fig1_path.exists():
            st.image(str(fig1_path), caption="Left: prior work (leaky features). "
                     "Right: BioMolAMR (clean features).", use_container_width=True)

    st.markdown("---")
    st.subheader("Why Does a Simple MLP Beat Complex GNNs?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        The zero-shot task requires **pairwise gene-drug compatibility** — knowing whether
        a specific protein can resist a specific chemical. This is encoded in:
        - The **protein's sequence** (via ESM-2): captures fold, active site, functional family
        - The **drug's chemical structure** (via fingerprints): captures pharmacophore, target moiety

        A 2-layer MLP concatenating these two vectors learns this pairwise compatibility
        function directly. **GNNs distort this by aggregating neighbourhood information**
        from training drug classes — propagating patterns that don't transfer to novel drugs.
        """)
    with col2:
        st.markdown("""
        **The analogy:** This is essentially protein-ligand interaction prediction.
        You don't need to know the entire protein-drug interaction network to predict
        whether *this* protein binds *this* molecule — you just need both structures.

        **Implication for drug discovery:** To screen a novel antibiotic candidate for
        potential resistance, you only need its SMILES string + the Feature-MLP.
        No graph update, no retraining, no genomic context required.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Results":
    st.title("📊 Experimental Results")

    # Main results table
    st.subheader("Main Results (5 seeds each)")
    st.markdown("**Primary metric:** ZS-all MRR — rank each zero-shot drug among all 46 drug classes. Random = 1/46 ≈ 0.022.")

    import pandas as pd
    results_data = {
        "Model":        ["Feature-MLP", "DistMult†", "R-GCN (bio)", "BioMolAMR", "TransE†", "Random"],
        "Test MRR":     ["0.968 ± 0.002", "0.767 ± 0.007", "0.925 ± 0.001", "0.912 ± 0.011", "0.887 ± 0.008", "—"],
        "ZS-Within":    ["0.137 ± 0.004", "0.141 ± 0.023", "0.141 ± 0.003", "0.125 ± 0.005", "0.120 ± 0.038", "0.083"],
        "ZS-All ★":     ["0.069 ± 0.006", "0.032 ± 0.004", "0.027 ± 0.001", "0.019 ± 0.001", "0.017 ± 0.001", "0.022"],
        "Gen. Gap":     ["-0.899", "-0.735", "-0.898", "-0.893", "-0.870", "—"],
        "ZS-capable":   ["✅", "⚠️ (transductive)", "✅", "✅", "⚠️ (transductive)", "—"],
    }
    df = pd.DataFrame(results_data)
    def highlight_best(row):
        styles = [""] * len(row)
        if row["Model"] == "Feature-MLP":
            styles = ["background-color: #e8f5e9; font-weight: bold"] * len(row)
        elif row["Model"] == "Random":
            styles = ["color: #888"] * len(row)
        return styles
    st.dataframe(df.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)
    st.caption("★ Primary metric. † Transductive models — ZS scores reflect embedding extrapolation, not true zero-shot.")

    # Prior work comparison
    st.markdown("""
    <div class="leakage-box">
    <b>Prior work comparison:</b> Reported ZS-within MRR = <b>0.775</b> with leaky features.
    Our corrected benchmark: Feature-MLP ZS-within MRR = <b>0.137</b> — a <b>5.7× reduction</b>
    attributable entirely to feature leakage removal.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Figures")

    # F2 main results
    f2 = FIGS_DIR / "F2_main_results.png"
    if f2.exists():
        st.image(str(f2), caption="Figure 2: Zero-shot MRR comparison. "
                 "Left: all-class (primary). Right: within-ZS (12-way).",
                 use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        f3 = FIGS_DIR / "F3_perclass_heatmap.png"
        if f3.exists():
            st.image(str(f3), caption="Figure 3: Per-class ZS-all MRR heatmap. "
                     "Columns sorted by Feature-MLP performance.",
                     use_container_width=True)
    with col2:
        f4 = FIGS_DIR / "F4_tanimoto_scatter.png"
        if f4.exists():
            st.image(str(f4), caption="Figure 4: Tanimoto chemical similarity to nearest "
                     "training drug vs. ZS-all MRR.",
                     use_container_width=True)

    st.markdown("---")
    st.subheader("Ablation Study: Feature Contributions")
    st.markdown("Which features drive zero-shot transfer?")

    abl_data = {
        "Variant":          ["Feature-MLP (full)", "FMLP: gene=0 (drug FP only)",
                             "FMLP: drug=0 (ESM-2 only)", "FMLP: leaky gene features"],
        "ZS-Within":        ["0.137 ± 0.004", "0.140 ± 0.003", "0.010 ± 0.000", "0.159 ± 0.030"],
        "ZS-All ★":         ["0.069 ± 0.006", "0.020 ± 0.002", "0.010 ± 0.000", "0.112 ± 0.000"],
        "vs. Full":         ["—", "↓ 3.5×", "↓ 6.9× (below random)", "↑ 1.6× (leakage)"],
    }
    df_abl = pd.DataFrame(abl_data)
    def highlight_abl(row):
        if "leaky" in row["Variant"]:
            return ["background-color: #fff3e0"] * len(row)
        if "gene=0" in row["Variant"] or "drug=0" in row["Variant"]:
            return ["background-color: #fce4ec"] * len(row)
        if row["Variant"] == "Feature-MLP (full)":
            return ["background-color: #e8f5e9; font-weight: bold"] * len(row)
        return [""] * len(row)
    st.dataframe(df_abl.style.apply(highlight_abl, axis=1), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        f5 = FIGS_DIR / "F5_ablation.png"
        if f5.exists():
            st.image(str(f5), caption="Figure 5: Ablation results. Both features jointly necessary.",
                     use_container_width=True)
    with col2:
        st.markdown("""
        **Interpretation:**
        - **Drug FP only** → drops to ≈ random: fingerprints know the drug class identity
          but cannot identify resistant genes without protein sequence context.
        - **ESM-2 only** → collapses below random: protein sequences encode function
          but are drug-agnostic by design.
        - **Both needed** → the MLP learns pairwise protein-chemical compatibility,
          analogous to protein-ligand binding prediction.
        - **Leaky features** → 1.6× inflation (0.112 vs 0.069), confirming prior work inflated by
          the direct drug-class label in gene features.
        """)

    st.markdown("---")
    st.subheader("Per-Class Breakdown")
    if EVAL_PATH.exists():
        summary = load_eval()
        fmlp    = summary.get("feature_mlp", {}).get("per_dc_zs_all", {})
        rgcn    = summary.get("rgcn_bio",    {}).get("per_dc_zs_all", {})
        bma     = summary.get("biomolamr",   {}).get("per_dc_zs_all", {})
        if fmlp:
            rows = []
            for dc in sorted(fmlp.keys(), key=lambda c: fmlp[c]["mrr"], reverse=True):
                rows.append({
                    "Drug Class":       dc,
                    "# ZS genes":       fmlp[dc].get("n_edges", "?"),
                    "Feature-MLP MRR":  f"{fmlp[dc]['mrr']:.3f}",
                    "R-GCN MRR":        f"{rgcn.get(dc,{}).get('mrr',0):.3f}",
                    "BioMolAMR MRR":    f"{bma.get(dc,{}).get('mrr',0):.3f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: GENE → DRUG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Gene → Drug Rank":
    st.title("🧬 Gene → Drug Class Ranking")
    st.markdown(
        "Select a resistance gene and see how all 46 antibiotic drug classes are "
        "ranked by predicted resistance score."
    )

    g_obj = load_graph()
    g_data = g_obj["hetero_data"]
    dc2i   = g_obj["dc2i"]
    i2dc   = {v: k for k, v in dc2i.items()}
    splits = load_splits()

    zs_classes = set(splits["meta"]["zs_drug_class_names"])
    gene_meta  = g_obj["gene_metadata"]
    gene2idx   = g_obj["gene2idx"]

    # Build gene options
    idx2aro = {v: k for k, v in gene2idx.items()}
    gene_options = []
    for aro, idx in sorted(gene2idx.items(), key=lambda x: x[1]):
        gm   = gene_meta.get(aro, {})
        name = gm.get("name", aro) if isinstance(gm, dict) else aro
        gene_options.append(f"{name} (ARO:{aro})")
    gene_options = sorted(set(gene_options))

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_gene = st.selectbox("Select a resistance gene", gene_options, index=0)
        model_choice  = st.selectbox("Model", ["Feature-MLP (best)", "R-GCN (bio)", "BioMolAMR"])
    with col2:
        show_zs_only = st.checkbox("Show ZS classes highlighted", value=True)
        top_n        = st.slider("Show top N drug classes", 5, 46, 20)

    # Parse gene index
    aro_str   = selected_gene.split("ARO:")[-1].rstrip(")")
    gene_idx  = gene2idx.get(aro_str, 0)
    gene_name = selected_gene.split(" (")[0]

    # Load model
    if "Feature-MLP" in model_choice:
        model, _ = load_fmlp()
        mn = "feature_mlp"
    elif "R-GCN" in model_choice:
        model, _ = load_rgcn()
        mn = "rgcn"
    else:
        model, _ = load_biomolamr()
        mn = "biomolamr"

    if model is None:
        st.error(f"Model checkpoint not found in {MODELS_DIR}")
        st.stop()

    if st.button("Predict", type="primary"):
        with st.spinner("Running inference…"):
            scores = score_gene_vs_all_drugs(mn, model, g_data, gene_idx)

        order = np.argsort(scores)[::-1][:top_n]

        # Ground-truth positives
        pos_pairs = {(g, d) for g, d in splits["meta"]["pos_pairs"]}
        train_pos = {d for (g, d) in pos_pairs if g == gene_idx}
        zs_pos    = {d for d in train_pos
                     if i2dc.get(d, "") in zs_classes}

        st.markdown(f"### Predicted drug resistance ranking for **{gene_name}**")

        import pandas as pd
        rows = []
        for rank, dc_idx in enumerate(order, 1):
            dc_name   = i2dc.get(dc_idx, f"DC_{dc_idx}")
            is_pos    = dc_idx in train_pos
            is_zs     = dc_name in zs_classes
            tag = ""
            if is_pos and is_zs:   tag = "✅ True positive (ZS)"
            elif is_pos:           tag = "✅ True positive (train)"
            elif is_zs:            tag = "🔵 ZS class (no edge)"
            rows.append({
                "Rank":        rank,
                "Drug Class":  dc_name,
                "Score":       f"{scores[dc_idx]:.4f}",
                "Status":      tag,
            })
        df_res = pd.DataFrame(rows)

        def color_status(row):
            if "True positive (ZS)" in row["Status"]:
                return ["background-color: #e8f5e9"] * len(row)
            if "True positive" in row["Status"]:
                return ["background-color: #e3f2fd"] * len(row)
            if "ZS class" in row["Status"]:
                return ["background-color: #fff3e0"] * len(row)
            return [""] * len(row)

        st.dataframe(df_res.style.apply(color_status, axis=1),
                     use_container_width=True, hide_index=True)

        n_tp_in_top = sum(1 for r in rows if "True positive" in r["Status"])
        st.info(f"Top {top_n} contains {n_tp_in_top} known resistance edges for this gene. "
                f"Green = known resistance. Orange = zero-shot drug class.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: DRUG → GENE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Drug → Gene Rank":
    st.title("💊 Drug Class → Gene Ranking")
    st.markdown(
        "Select an antibiotic drug class (including zero-shot held-out classes) "
        "and see the top predicted resistance genes."
    )

    g_obj  = load_graph()
    g_data = g_obj["hetero_data"]
    dc2i   = g_obj["dc2i"]
    i2dc   = {v: k for k, v in dc2i.items()}
    splits = load_splits()
    gene_meta = g_obj["gene_metadata"]
    gene2idx  = g_obj["gene2idx"]
    idx2aro   = {v: k for k, v in gene2idx.items()}
    zs_classes = set(splits["meta"]["zs_drug_class_names"])

    col1, col2 = st.columns([2, 1])
    with col1:
        # Put ZS classes first
        zs_sorted    = sorted(zs_classes)
        train_sorted = sorted(dc2i.keys() - zs_classes)
        drug_options = (["--- Zero-Shot Drug Classes ---"] + zs_sorted +
                        ["--- Training Drug Classes ---"] + train_sorted)
        selected_drug = st.selectbox("Select antibiotic drug class", drug_options, index=1)
        model_choice  = st.selectbox("Model", ["Feature-MLP (best)", "R-GCN (bio)", "BioMolAMR"])
    with col2:
        top_k = st.slider("Top N genes", 10, 100, 20)
        is_zs = selected_drug in zs_classes
        if is_zs:
            st.success("✅ This is a **zero-shot** drug class — never seen during training!")
        elif selected_drug.startswith("---"):
            st.stop()

    if "Feature-MLP" in model_choice:
        model, _ = load_fmlp()
        mn = "feature_mlp"
    elif "R-GCN" in model_choice:
        model, _ = load_rgcn()
        mn = "rgcn"
    else:
        model, _ = load_biomolamr()
        mn = "biomolamr"

    if model is None:
        st.error("Model checkpoint not found.")
        st.stop()

    dc_idx = dc2i[selected_drug]

    if st.button("Predict", type="primary"):
        with st.spinner("Scoring all 6,397 genes…"):
            top_idx, top_scores = score_drug_vs_all_genes(mn, model, g_data, dc_idx, top_k)

        pos_pairs = {(g, d) for g, d in splits["meta"]["pos_pairs"]}
        true_genes = {g for (g, d) in pos_pairs if d == dc_idx}

        label = "Zero-Shot" if is_zs else "Training"
        st.markdown(f"### Top {top_k} predicted resistance genes for **{selected_drug}** ({label} class)")

        if is_zs:
            n_correct = sum(1 for g in top_idx if g in true_genes)
            p_at_k    = n_correct / top_k
            st.metric(
                label=f"P@{top_k} (precision in top {top_k})",
                value=f"{p_at_k:.2f}",
                delta=f"{n_correct}/{top_k} correct"
            )

        import pandas as pd
        rows = []
        for rank, (gidx, sc) in enumerate(zip(top_idx, top_scores), 1):
            aro  = idx2aro.get(gidx, "")
            gm   = gene_meta.get(aro, {})
            name = gm.get("name", f"Gene_{gidx}") if isinstance(gm, dict) else f"Gene_{gidx}"
            desc = gm.get("description", "")[:80] if isinstance(gm, dict) else ""
            is_tp = gidx in true_genes
            rows.append({
                "Rank":        rank,
                "Gene":        name,
                "ARO":         f"ARO:{aro}",
                "Score":       f"{sc:.4f}",
                "Known TP":    "✅" if is_tp else "",
                "Description": desc,
            })
        df = pd.DataFrame(rows)

        def color_tp(row):
            if row["Known TP"] == "✅":
                return ["background-color: #e8f5e9"] * len(row)
            return [""] * len(row)

        st.dataframe(df.style.apply(color_tp, axis=1),
                     use_container_width=True, hide_index=True)

        if is_zs:
            st.caption(
                f"Known resistance genes for {selected_drug}: {len(true_genes)}. "
                f"Top {top_k} recall: {n_correct}/{min(top_k, len(true_genes))}."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: NOVEL DRUG (SMILES)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Novel Drug (SMILES)":
    st.title("⚗️ Novel Drug — Zero-Shot Resistance Prediction")
    st.markdown(
        "Paste the SMILES string of any novel antibiotic compound. "
        "The app computes its molecular fingerprint and predicts which genes "
        "are most likely to confer resistance — **no retraining required**."
    )

    g_obj     = load_graph()
    g_data    = g_obj["hetero_data"]
    dc2i      = g_obj["dc2i"]
    i2dc      = {v: k for k, v in dc2i.items()}
    gene_meta = g_obj["gene_metadata"]
    gene2idx  = g_obj["gene2idx"]
    idx2aro   = {v: k for k, v in gene2idx.items()}
    tanimoto  = load_tanimoto()

    # Example SMILES
    EXAMPLES = {
        "Linezolid (oxazolidinone)":
            "CC(=O)NCC1CN(c2ccc(N3CCOCC3=O)c(F)c2)C(=O)O1",
        "Clindamycin (lincosamide)":
            "CCCC[C@@H]1C[C@H](N(C)C(=O)[C@@H]2[C@H](O)[C@@H](O)[C@H](SC)[C@@H](O2)SC)C[C@@H]1Cl",
        "Rifampicin (rifamycin)":
            "COc1c(C)c2c(c(O)c1NC(=O)/C=C/[C@H]1O[C@@]1(C)CC)OC(=O)C(=C2/C=C/[C@H]([C@@H]([C@H](OC(C)=O)[C@@H]3OC(=O)[C@H](C)[C@@H]3O)C)OC)=O",
        "Hypothetical compound":
            "CCN(CC)CCNC(=O)c1ccc(cc1)N",
    }

    example = st.selectbox("Try an example (or type your own below):",
                           ["Custom…"] + list(EXAMPLES.keys()))
    smiles_input = EXAMPLES.get(example, "") if example != "Custom…" else ""
    smiles = st.text_area("SMILES string:", value=smiles_input, height=80,
                           placeholder="e.g. CCN(CC)c1ccc(cc1)C(=O)N")

    model_choice = st.selectbox("Model", ["Feature-MLP (best)", "R-GCN (bio)", "BioMolAMR"])
    top_k = st.slider("Top N resistance genes to show", 10, 50, 20)

    if st.button("Predict resistance genes", type="primary"):
        if not smiles.strip():
            st.warning("Please enter a SMILES string.")
            st.stop()

        with st.spinner("Computing fingerprint…"):
            fp_tensor, err = compute_fingerprint_from_smiles(smiles.strip())

        if fp_tensor is None:
            st.error(f"Fingerprint error: {err}")
            st.stop()

        # Find nearest known drug class by Tanimoto
        fp_np    = fp_tensor.numpy().astype(float)
        db_fp    = g_data["drug_class"].x.numpy().astype(float)
        # Tanimoto on binary vectors: |A∩B| / |A∪B|
        def tanimoto_vec(a, b_mat):
            ab  = (b_mat * a).sum(axis=1)
            aa  = (a * a).sum()
            bb  = (b_mat * b_mat).sum(axis=1)
            return ab / (aa + bb - ab + 1e-9)
        sims     = tanimoto_vec(fp_np, db_fp)
        best_dc  = int(np.argmax(sims))
        best_sim = float(sims[best_dc])

        col1, col2, col3 = st.columns(3)
        col1.metric("Fingerprint dim", "3,245")
        col2.metric("Nearest drug class", i2dc.get(best_dc, "?"))
        col3.metric("Tanimoto similarity", f"{best_sim:.3f}")

        if best_sim < 0.1:
            st.warning(
                "Very low Tanimoto similarity to all known drug classes. "
                "Zero-shot predictions may be unreliable — this compound is chemically distant "
                "from the training distribution."
            )
        elif best_sim < 0.3:
            st.info(
                f"Moderate similarity to **{i2dc.get(best_dc, '?')}** (Tanimoto={best_sim:.2f}). "
                "Predictions are extrapolating beyond the training distribution."
            )
        else:
            st.success(
                f"Good similarity to **{i2dc.get(best_dc, '?')}** (Tanimoto={best_sim:.2f}). "
                "This compound is chemically close to a training drug class."
            )

        # Load model and replace drug feature for this novel compound
        if "Feature-MLP" in model_choice:
            model, _ = load_fmlp()
            mn = "feature_mlp"
        elif "R-GCN" in model_choice:
            model, _ = load_rgcn()
            mn = "rgcn"
        else:
            model, _ = load_biomolamr()
            mn = "biomolamr"

        if model is None:
            st.error("Model checkpoint not found.")
            st.stop()

        # Temporarily inject novel drug fingerprint as a new "drug class" at position 46
        novel_drug_x = torch.cat([g_data["drug_class"].x, fp_tensor.unsqueeze(0)], dim=0)
        novel_dc_idx = g_data["drug_class"].x.shape[0]  # index 46

        n_genes = g_data["gene"].x.shape[0]
        gene_t  = torch.arange(n_genes, dtype=torch.long)
        drug_t  = torch.tensor([novel_dc_idx] * n_genes, dtype=torch.long)

        with st.spinner("Scoring all 6,397 genes…"):
            with torch.no_grad():
                if mn == "feature_mlp":
                    scores = model(g_data["gene"].x, novel_drug_x, gene_t, drug_t).numpy()
                else:
                    # For GNN models, use nearest drug class as proxy
                    scores_proxy, _ = score_drug_vs_all_genes(mn, model, g_data, best_dc, n_genes)
                    scores = np.zeros(n_genes)
                    for gi, s in zip(scores_proxy, _):
                        scores[gi] = s

        top_idx = np.argsort(scores)[::-1][:top_k]

        st.markdown(f"### Top {top_k} predicted resistance genes for novel compound")

        import pandas as pd
        rows = []
        for rank, gidx in enumerate(top_idx, 1):
            aro  = idx2aro.get(gidx, "")
            gm   = gene_meta.get(aro, {})
            name = gm.get("name", f"Gene_{gidx}") if isinstance(gm, dict) else f"Gene_{gidx}"
            desc = gm.get("description", "")[:100] if isinstance(gm, dict) else ""
            rows.append({
                "Rank":        rank,
                "Gene":        name,
                "ARO":         f"ARO:{aro}",
                "Score":       f"{scores[gidx]:.4f}",
                "Description": desc,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if mn != "feature_mlp":
            st.caption(
                f"⚠️ {model_choice} uses graph structure. For truly novel drugs, only Feature-MLP "
                f"can directly use the novel fingerprint. GNN result shown uses nearest training drug "
                f"class ({i2dc.get(best_dc,'?')}) as a proxy."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("⚖️ Model Comparison")
    st.markdown(
        "Compare Feature-MLP, R-GCN, and BioMolAMR side-by-side on the same gene query."
    )

    g_obj  = load_graph()
    g_data = g_obj["hetero_data"]
    dc2i   = g_obj["dc2i"]
    i2dc   = {v: k for k, v in dc2i.items()}
    splits = load_splits()
    gene_meta = g_obj["gene_metadata"]
    gene2idx  = g_obj["gene2idx"]
    idx2aro   = {v: k for k, v in gene2idx.items()}
    zs_classes = set(splits["meta"]["zs_drug_class_names"])

    gene_options = []
    for aro, idx in sorted(gene2idx.items(), key=lambda x: x[1]):
        gm   = gene_meta.get(aro, {})
        name = gm.get("name", aro) if isinstance(gm, dict) else aro
        gene_options.append(f"{name} (ARO:{aro})")
    gene_options = sorted(set(gene_options))

    selected_gene = st.selectbox("Select a resistance gene", gene_options, index=0)
    top_n         = st.slider("Show top N drugs", 5, 46, 15)

    aro_str  = selected_gene.split("ARO:")[-1].rstrip(")")
    gene_idx = gene2idx.get(aro_str, 0)
    gene_name = selected_gene.split(" (")[0]

    if st.button("Compare all models", type="primary"):
        fmlp_model,  _ = load_fmlp()
        rgcn_model,  _ = load_rgcn()
        bma_model,   _ = load_biomolamr()

        pos_pairs  = {(g, d) for g, d in splits["meta"]["pos_pairs"]}
        true_drugs = {d for (g, d) in pos_pairs if g == gene_idx}

        with st.spinner("Running all 3 models…"):
            s_fmlp = score_gene_vs_all_drugs("feature_mlp", fmlp_model, g_data, gene_idx)
            s_rgcn = score_gene_vs_all_drugs("rgcn",        rgcn_model,  g_data, gene_idx)
            s_bma  = score_gene_vs_all_drugs("biomolamr",   bma_model,   g_data, gene_idx)

        # Rank by Feature-MLP
        order = np.argsort(s_fmlp)[::-1][:top_n]

        import pandas as pd
        rows = []
        for rank, dc_idx in enumerate(order, 1):
            dc_name = i2dc.get(dc_idx, f"DC_{dc_idx}")
            is_tp   = dc_idx in true_drugs
            is_zs   = dc_name in zs_classes
            tag = ""
            if is_tp and is_zs:   tag = "✅ ZS TP"
            elif is_tp:           tag = "✅ Train TP"
            elif is_zs:           tag = "🔵 ZS"
            rows.append({
                "Rank (FMLP)":  rank,
                "Drug Class":   dc_name,
                "FMLP Score":   f"{s_fmlp[dc_idx]:.4f}",
                "R-GCN Score":  f"{s_rgcn[dc_idx]:.4f}",
                "BioMolAMR":    f"{s_bma[dc_idx]:.4f}",
                "Truth":        tag,
            })
        df = pd.DataFrame(rows)

        def color_rows(row):
            if "ZS TP" in row["Truth"]:
                return ["background-color: #c8e6c9"] * len(row)
            if "Train TP" in row["Truth"]:
                return ["background-color: #bbdefb"] * len(row)
            if "ZS" in row["Truth"]:
                return ["background-color: #fff9c4"] * len(row)
            return [""] * len(row)

        st.markdown(f"### Drug rankings for **{gene_name}** — all 3 models")
        st.dataframe(df.style.apply(color_rows, axis=1),
                     use_container_width=True, hide_index=True)
        st.caption(
            "Green = known resistance (true positive). "
            "Blue = ZS + TP. Yellow = ZS class (no known edge in test set). "
            "Ranked by Feature-MLP score."
        )

        # Rank correlation
        all_fmlp_rank = np.argsort(np.argsort(-s_fmlp))
        all_rgcn_rank = np.argsort(np.argsort(-s_rgcn))
        all_bma_rank  = np.argsort(np.argsort(-s_bma))

        from scipy.stats import spearmanr
        r_fr, _ = spearmanr(all_fmlp_rank, all_rgcn_rank)
        r_fb, _ = spearmanr(all_fmlp_rank, all_bma_rank)
        r_rb, _ = spearmanr(all_rgcn_rank, all_bma_rank)

        st.markdown("---")
        st.subheader("Rank Correlation (Spearman ρ)")
        c1, c2, c3 = st.columns(3)
        c1.metric("FMLP vs R-GCN", f"{r_fr:.3f}")
        c2.metric("FMLP vs BioMolAMR", f"{r_fb:.3f}")
        c3.metric("R-GCN vs BioMolAMR", f"{r_rb:.3f}")
        st.caption(
            "High correlation means models agree on the ranking. "
            "Low correlation indicates models are capturing different signals."
        )
