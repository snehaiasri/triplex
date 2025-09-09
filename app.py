# app.py — Streamlit 3-model pipeline (Python 3.9 compatible)

import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# ----------------------------
# Page config & header
# ----------------------------
st.set_page_config(page_title="TriPLex", layout="wide", initial_sidebar_state="expanded")

c1, c2, c3 = st.columns([1.5, 20, 2])
with c1:
    st.image("static/images/icarlogo.png", width=120)
with c2:
    st.markdown("<h1 style='text-align:center;'>TriPLex: Tri-Stage Protein–Ligand Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center; margin-top:-10px;'>Bioactivity ➜ Interaction ➜ Binding Affinity</h5>", unsafe_allow_html=True)
with c3:
    st.image("static/images/iasri-logo.png", width=120)

st.markdown("---")

# ----------------------------
# Model artefact paths (adjust to your layout)
# ----------------------------
PATH_MODEL1 = "models/model1"         # bioactivity_model.pkl, scaler.pkl, descriptor_columns.txt
PATH_MODEL2 = "models/model2"         # interaction_model.pkl, scaler.pkl, meta.json
PATH_MODEL3 = "models/model3"         # affinity_q_low.pkl, affinity_q_med.pkl, affinity_q_high.pkl, scaler.pkl, meta.json

# ----------------------------
# Session State
# ----------------------------
ss = st.session_state
if "model1_df" not in ss: ss.model1_df = None
if "model1_active_df" not in ss: ss.model1_active_df = None
if "model2_df" not in ss: ss.model2_df = None

# ----------------------------
# Helpers
# ----------------------------
def df_to_download_bytes(df: pd.DataFrame, filename: str = "results.csv"):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode(), filename

def safe_read_csv(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None

# ----------------------------
# MODEL 1: Bioactivity (compound-only)
# ----------------------------
def load_model1():
    model = joblib.load(f"{PATH_MODEL1}/bioactivity_model.pkl")
    scaler = joblib.load(f"{PATH_MODEL1}/scaler.pkl")
    with open(f"{PATH_MODEL1}/descriptor_columns.txt") as f:
        descriptor_names = [ln.strip() for ln in f if ln.strip()]
    descriptor_funcs = {
        "MolWt": Descriptors.MolWt,
        "MolLogP": Descriptors.MolLogP,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "TPSA": Descriptors.TPSA,
        "NumRotatableBonds": Descriptors.NumRotatableBonds
    }
    return model, scaler, descriptor_names, descriptor_funcs

def compute_rdkit_desc(smiles: str, descriptor_names, descriptor_funcs):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return [descriptor_funcs[name](mol) for name in descriptor_names]
    except KeyError as e:
        st.error(f"Descriptor {e} not found. Check descriptor_columns.txt.")
        return None

# ----------------------------
# MODEL 2 & 3: Shared featurization
# ----------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFYW")
POLAR       = set("STNQCY")
CHARGED     = set("KRHDE")

def aa_composition(seq: str) -> np.ndarray:
    s = (seq or "").upper()
    counts = np.array([s.count(a) for a in AMINO_ACIDS], dtype=float)
    total = max(len(s), 1)
    return counts / total

def protein_basic_props(seq: str) -> np.ndarray:
    s = (seq or "").upper()
    L = float(len(s))
    if L == 0:
        return np.zeros(4, dtype=float)
    hyd = sum(1 for c in s if c in HYDROPHOBIC) / L
    pol = sum(1 for c in s if c in POLAR) / L
    chg = sum(1 for c in s if c in CHARGED) / L
    return np.array([L, hyd, pol, chg], dtype=float)

def featurize_protein(seq: str) -> np.ndarray:
    return np.concatenate([aa_composition(seq), protein_basic_props(seq)], axis=0)  # 24 dims

def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> Optional[np.ndarray]:
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def featurize_pair(seq: str, smiles: str, n_bits: int = 2048, radius: int = 2):
    lig = morgan_fp(smiles, n_bits=n_bits, radius=radius)
    if lig is None:
        return None
    prot = featurize_protein(seq)
    return np.concatenate([prot, lig], axis=0)

# ----------------------------
# MODEL 2: Interaction Classifier
# ----------------------------
def load_model2():
    meta = json.load(open(f"{PATH_MODEL2}/meta.json"))
    model = joblib.load(f"{PATH_MODEL2}/interaction_model.pkl")
    scaler = joblib.load(f"{PATH_MODEL2}/scaler.pkl")
    n_bits = int(meta["n_bits"]); radius = int(meta["radius"])
    threshold = float(meta.get("threshold", 0.5))
    return model, scaler, n_bits, radius, threshold

def predict_model2(df_pairs: pd.DataFrame, model, scaler, n_bits: int, radius: int, threshold: float = 0.5):
    need = ["protein_id","protein_sequence","compound_id","smiles"]
    for c in need:
        if c not in df_pairs.columns:
            raise ValueError(f"Missing column '{c}' for Model-2.")
    X, ok = [], []
    for _, r in df_pairs.iterrows():
        x = featurize_pair(r["protein_sequence"], r["smiles"], n_bits=n_bits, radius=radius)
        ok.append(x is not None)
        X.append(x if x is not None else np.zeros(24 + n_bits, dtype=float))
    X = np.vstack(X); ok = np.array(ok, dtype=bool)
    Xs = scaler.transform(X)
    proba1 = model.predict_proba(Xs)[:, 1]
    pred = (proba1 >= threshold).astype(int)

    out = df_pairs.copy()
    out["valid_smiles"] = ok.astype(int)
    out["interaction_prob"] = proba1
    out["interaction_pred"] = pred
    return out

# ----------------------------
# MODEL 3: Affinity Regressor
# ----------------------------
def load_model3():
    meta = json.load(open(f"{PATH_MODEL3}/meta.json"))
    n_bits = int(meta["n_bits"]); radius = int(meta["radius"])
    q_low = float(meta["q_low"]); q_high = float(meta["q_high"])
    m_low  = joblib.load(f"{PATH_MODEL3}/affinity_q_low.pkl")
    m_med  = joblib.load(f"{PATH_MODEL3}/affinity_q_med.pkl")
    m_high = joblib.load(f"{PATH_MODEL3}/affinity_q_high.pkl")
    scaler = joblib.load(f"{PATH_MODEL3}/scaler.pkl")
    return m_low, m_med, m_high, scaler, n_bits, radius, q_low, q_high

def predict_model3(df_pairs: pd.DataFrame,
                   m_low, m_med, m_high, scaler,
                   n_bits: int, radius: int,
                   q_low: float, q_high: float,
                   strong_thresh: Optional[float] = None):
    need = ["protein_id","protein_sequence","compound_id","smiles"]
    for c in need:
        if c not in df_pairs.columns:
            raise ValueError(f"Missing column '{c}' for Model-3.")
    X, ok = [], []
    for _, r in df_pairs.iterrows():
        x = featurize_pair(r["protein_sequence"], r["smiles"], n_bits=n_bits, radius=radius)
        ok.append(x is not None)
        X.append(x if x is not None else np.zeros(24 + n_bits, dtype=float))
    X = np.vstack(X); ok = np.array(ok, dtype=bool)
    Xs = scaler.transform(X)
    yL = m_low.predict(Xs)
    yM = m_med.predict(Xs)
    yU = m_high.predict(Xs)

    out = df_pairs.copy()
    out["valid_smiles"] = ok.astype(int)
    out["affinity_pred"] = yM
    out[f"affinity_q{int(q_low*100)}"]  = yL
    out[f"affinity_q{int(q_high*100)}"] = yU

    if strong_thresh is not None:
        iqr = np.maximum(yU - yL, 1e-6)
        z = (yM - strong_thresh) / (iqr / 1.349)  # IQR->sigma approx
        out["prob_strong_binding"] = 1.0 / (1.0 + np.exp(-z))
    return out

# =========================================================
# UI: Tabs for each model + Pipeline
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Model-1: Bioactivity (Compound)",
    "Model-2: Interaction (Protein–Ligand)",
    "Model-3: Binding Affinity",
    "Pipeline: 1 ➜ 2 ➜ 3"
])

# ----------------------------
# TAB 1
# ----------------------------
with tab1:
    st.subheader("Predict Bioactivity of Compounds (Active/Inactive + Probabilities)")
    try:
        with open("static/example.csv", "rb") as fh:
            st.download_button("📥 Download Sample File (Model-1)", fh.read(), file_name="example_model1.csv", mime="text/csv")
    except Exception:
        st.info("Place a sample CSV at static/example.csv with columns: molecule_id, SMILES")

    target = st.selectbox("Select Target Organism", ["Bacteria", "Fungi", "Virus", "Plant"])
    up1 = st.file_uploader("Upload CSV with columns: molecule_id, SMILES", type=["csv"], key="up1")

    if st.button("Run Model-1"):
        if up1 is None:
            st.warning("Please upload a CSV.")
        else:
            model1_df_in = safe_read_csv(up1)
            if model1_df_in is not None:
                model, scaler, descriptor_names, descriptor_funcs = load_model1()
                results = []
                for i, row in model1_df_in.iterrows():
                    smi = row.get("SMILES", None)
                    desc = compute_rdkit_desc(smi, descriptor_names, descriptor_funcs)
                    if desc:
                        scaled = scaler.transform([desc])
                        prob = model.predict_proba(scaled)[0]  # [prob_inactive, prob_active]
                        pred = model.predict(scaled)[0]
                        results.append({
                            "S.No.": i+1,
                            "molecule_id": row.get("molecule_id", i+1),
                            "SMILES": smi,
                            "Probability (Active)": round(float(prob[1]), 4),
                            "Probability (Inactive)": round(float(prob[0]), 4),
                            "Prediction": "Yes" if int(pred) == 1 else "No"
                        })
                    else:
                        results.append({
                            "S.No.": i+1,
                            "molecule_id": row.get("molecule_id", i+1),
                            "SMILES": smi,
                            "Probability (Active)": "Invalid",
                            "Probability (Inactive)": "Invalid",
                            "Prediction": "Invalid SMILES"
                        })
                out_df = pd.DataFrame(results)
                st.dataframe(out_df, use_container_width=True)
                ss.model1_df = out_df
                ss.model1_active_df = out_df[out_df["Prediction"] == "Yes"][["molecule_id","SMILES"]].reset_index(drop=True)

                data_bytes, fname = df_to_download_bytes(out_df, "bioactivity_predictions.csv")
                st.download_button("⬇️ Download Model-1 Results", data_bytes, file_name=fname, mime="text/csv")

# ----------------------------
# TAB 2
# ----------------------------
with tab2:
    st.subheader("Predict Protein–Ligand Interaction (Probability)")
    st.caption("Input CSV columns: protein_id, protein_sequence, compound_id, smiles")
    template2 = "protein_id,protein_sequence,compound_id,smiles\nP12345,MEKVL...KQL,ChEMBL1,CCO...\n"
    st.download_button("📥 Download Input Template (Model-2)", template2.encode(), file_name="template_model2.csv", mime="text/csv")

    use_m1 = st.checkbox("Use active compounds from Model-1 (then upload only proteins & map)")
    up2 = st.file_uploader("Upload pairs CSV (protein_id, protein_sequence, compound_id, smiles)", type=["csv"], key="up2")

    map_info = st.expander("🔧 Build pairs from proteins × Model-1 actives")
    with map_info:
        st.write("If using Model-1 actives, upload a **proteins CSV** with columns: protein_id, protein_sequence.")
        up_prot = st.file_uploader("Upload proteins CSV", type=["csv"], key="prot_only")
        if use_m1 and (ss.model1_active_df is None or ss.model1_active_df.empty):
            st.warning("No active compounds found in Model-1 results yet.")

    if st.button("Run Model-2"):
        try:
            if use_m1:
                if up_prot is None:
                    st.warning("Upload proteins CSV to pair with Model-1 active compounds.")
                    st.stop()
                prot_df = safe_read_csv(up_prot)
                if prot_df is None:
                    st.stop()
                if ss.model1_active_df is None or ss.model1_active_df.empty:
                    st.warning("No Model-1 active compounds available.")
                    st.stop()
                prot_df = prot_df[["protein_id","protein_sequence"]].dropna()
                act = ss.model1_active_df.rename(columns={"molecule_id":"compound_id", "SMILES":"smiles"})
                prot_df["key"] = 1; act["key"] = 1
                pairs_df = prot_df.merge(act, on="key").drop(columns=["key"])
                df_in = pairs_df
            else:
                if up2 is None:
                    st.warning("Please upload pairs CSV.")
                    st.stop()
                df_in = safe_read_csv(up2)
                if df_in is None:
                    st.stop()

            model2, scaler2, nbits2, radius2, thr2 = load_model2()
            df_out = predict_model2(df_in, model2, scaler2, nbits2, radius2, threshold=thr2)
            st.dataframe(df_out, use_container_width=True)
            ss.model2_df = df_out

            data_bytes, fname = df_to_download_bytes(df_out, "interaction_predictions.csv")
            st.download_button("⬇️ Download Model-2 Results", data_bytes, file_name=fname, mime="text/csv")
        except Exception as e:
            st.error(f"Model-2 error: {e}")

# ----------------------------
# TAB 3
# ----------------------------
with tab3:
    st.subheader("Predict Binding Affinity (Median + Quantile Bands)")
    st.caption("Input CSV columns: protein_id, protein_sequence, compound_id, smiles")
    template3 = "protein_id,protein_sequence,compound_id,smiles\nP12345,MEKVL...KQL,ChEMBL1,CCO...\n"
    st.download_button("📥 Download Input Template (Model-3)", template3.encode(), file_name="template_model3.csv", mime="text/csv")

    use_m2 = st.checkbox("Use interacting pairs from Model-2 (filter by threshold in that model)")
    strong_thresh = st.number_input("Optional: strong binding threshold (e.g., pKd ≥ 7.0). Leave blank for none.", value=7.0, step=0.1)

    up3 = st.file_uploader("Upload pairs CSV for affinity prediction", type=["csv"], key="up3")

    if st.button("Run Model-3"):
        try:
            if use_m2:
                if ss.model2_df is None or ss.model2_df.empty:
                    st.warning("No Model-2 results available. Run Model-2 first or upload a CSV.")
                    st.stop()
                df_in = ss.model2_df[ss.model2_df["interaction_pred"] == 1][["protein_id","protein_sequence","compound_id","smiles"]].reset_index(drop=True)
                if df_in.empty:
                    st.warning("No interacting pairs found in Model-2 results.")
                    st.stop()
            else:
                if up3 is None:
                    st.warning("Please upload CSV.")
                    st.stop()
                df_in = safe_read_csv(up3)
                if df_in is None:
                    st.stop()

            m_low, m_med, m_high, scaler3, nbits3, radius3, ql, qh = load_model3()
            df_out = predict_model3(df_in, m_low, m_med, m_high, scaler3, nbits3, radius3,
                                    q_low=ql, q_high=qh,
                                    strong_thresh=float(strong_thresh) if strong_thresh else None)
            st.dataframe(df_out, use_container_width=True)

            data_bytes, fname = df_to_download_bytes(df_out, "affinity_predictions.csv")
            st.download_button("⬇️ Download Model-3 Results", data_bytes, file_name=fname, mime="text/csv")
        except Exception as e:
            st.error(f"Model-3 error: {e}")

# ----------------------------
# TAB 4 — Pipeline
# ----------------------------
with tab4:
    st.subheader("One-Click Pipeline")
    st.caption("Run: Model-1 (filter actives) ➜ build pairs with proteins ➜ Model-2 ➜ Model-3")

    cA, cB = st.columns([1,1])
    with cA:
        st.markdown("**Step-A: Upload compounds for Model-1** (molecule_id, SMILES)")
        upA = st.file_uploader("Compounds CSV", type=["csv"], key="pipe_cpd")
        st.markdown("**Step-B: Upload proteins** (protein_id, protein_sequence)")
        upB = st.file_uploader("Proteins CSV", type=["csv"], key="pipe_prot")

    with cB:
        st.markdown("**Options**")
        strong_thresh_pipe = st.number_input("Strong binding threshold for Model-3 (optional)", value=7.0, step=0.1)
        run_btn = st.button("Run Full Pipeline")

    if run_btn:
        try:
            comp_df = safe_read_csv(upA); prot_df = safe_read_csv(upB)
            if comp_df is None or prot_df is None:
                st.warning("Please upload both Compounds and Proteins CSVs.")
                st.stop()

            model1, scaler1, desc_names, desc_funcs = load_model1()
            m1_rows = []
            for i, r in comp_df.iterrows():
                smi = r.get("SMILES", None)
                desc = compute_rdkit_desc(smi, desc_names, desc_funcs)
                if desc:
                    scaled = scaler1.transform([desc])
                    prob = model1.predict_proba(scaled)[0]
                    pred = model1.predict(scaled)[0]
                    m1_rows.append({"molecule_id": r.get("molecule_id", i+1),
                                    "SMILES": smi, "p_active": float(prob[1]),
                                    "pred_active": int(pred)})
            m1_out = pd.DataFrame(m1_rows)
            actives = m1_out[m1_out["pred_active"] == 1][["molecule_id","SMILES"]].reset_index(drop=True)
            st.write("Model-1: Active compounds:", actives.shape[0])

            prot_df = prot_df[["protein_id","protein_sequence"]].dropna()
            actives = actives.rename(columns={"molecule_id":"compound_id", "SMILES":"smiles"})
            prot_df["key"] = 1; actives["key"] = 1
            pairs = prot_df.merge(actives, on="key").drop(columns=["key"])
            st.write("Pairs built:", pairs.shape[0])

            model2, scaler2, nbits2, radius2, thr2 = load_model2()
            m2_out = predict_model2(pairs, model2, scaler2, nbits2, radius2, threshold=thr2)
            inter = m2_out[m2_out["interaction_pred"] == 1][["protein_id","protein_sequence","compound_id","smiles"]].reset_index(drop=True)
            st.write("Model-2: Interacting pairs:", inter.shape[0])

            m_low, m_med, m_high, scaler3, nbits3, radius3, ql, qh = load_model3()
            m3_out = predict_model3(inter, m_low, m_med, m_high, scaler3, nbits3, radius3,
                                    q_low=ql, q_high=qh,
                                    strong_thresh=float(strong_thresh_pipe) if strong_thresh_pipe else None)

            st.markdown("### Pipeline Output (Model-3 predictions)")
            st.dataframe(m3_out, use_container_width=True)

            data_bytes, fname = df_to_download_bytes(m3_out, "pipeline_affinity_predictions.csv")
            st.download_button("⬇️ Download Pipeline Output", data_bytes, file_name=fname, mime="text/csv")
        except Exception as e:
            st.error(f"Pipeline error: {e}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div style='background-color:#32CD32; text-align:center'><p style='color:white'>© 2025 ICAR-Indian Agricultural Statistics Research Institute, New Delhi-110012</p></div>", unsafe_allow_html=True)
