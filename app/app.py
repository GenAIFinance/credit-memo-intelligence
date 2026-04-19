"""
app/app.py

Phase 6: Streamlit Demo UI
---------------------------
Four-page portfolio demo for the Credit Memo Intelligence Platform.
Each page is explicitly mapped to a skill in the DB AI Data Scientist
job description — see the skill badge on each page header.

Pages:
    1. Taxonomy Explorer   — LLM-first exploratory analysis
    2. Cluster Map         — Rapid semantic clustering
    3. Similar Memo Search — LangChain retrieval module
    4. Decomposition View  — Functional decomposition automation

Run:
    streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Memo Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark financial terminal aesthetic ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* Root variables */
:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --amber:         #f0a500;
    --amber-dim:     #8b6000;
    --text-primary:  #e6edf3;
    --text-muted:    #7d8590;
    --border:        #30363d;
    --green:         #3fb950;
    --red:           #f85149;
    --blue:          #58a6ff;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--amber);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Cards */
.memo-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-family: 'IBM Plex Sans', sans-serif;
}
.memo-card:hover { border-left-color: #f0c040; }

.metric-pill {
    display: inline-block;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--amber);
    margin-right: 6px;
    margin-bottom: 4px;
}

.score-badge {
    display: inline-block;
    background: var(--amber-dim);
    color: var(--amber);
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
}

.section-tag {
    display: inline-block;
    background: #1a2332;
    border: 1px solid #253a5e;
    color: var(--blue);
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 8px;
}

/* Skill badge */
.skill-badge {
    background: var(--bg-secondary);
    border: 1px solid var(--amber-dim);
    border-radius: 4px;
    padding: 6px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--amber);
    letter-spacing: 0.08em;
    display: inline-block;
    margin-bottom: 1.5rem;
}

/* Decomposition slots */
.slot-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    height: 100%;
}
.slot-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--amber);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.slot-text {
    font-size: 0.9rem;
    color: var(--text-primary);
    line-height: 1.6;
}

/* Page header */
.page-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}
.page-sub {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* Streamlit overrides */
.stSelectbox label, .stTextInput label, .stSlider label, .stMultiSelect label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
div[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
}
.stButton > button {
    background: var(--amber);
    color: #0d1117;
    border: none;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    border-radius: 4px;
}
.stButton > button:hover { background: #f0c040; }
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_chunks() -> pd.DataFrame:
    """Load chunk table from DuckDB."""
    db_path = ROOT / "data" / "processed" / "corpus.duckdb"
    if not db_path.exists():
        return pd.DataFrame()
    conn = duckdb.connect(str(db_path), read_only=True)
    df = conn.execute("SELECT * FROM chunks").df()
    conn.close()
    return df


@st.cache_data
def load_taxonomy() -> dict:
    """Load taxonomy.json from Phase 3."""
    path = ROOT / "data" / "processed" / "taxonomy.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_llm_labels() -> pd.DataFrame:
    """Load llm_labels.parquet from Phase 3."""
    path = ROOT / "data" / "processed" / "llm_labels.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_cluster_purity() -> dict:
    """Load cluster_purity.json from Phase 4."""
    path = ROOT / "data" / "processed" / "cluster_purity.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _data_missing_warning(label: str) -> None:
    st.warning(
        f"**{label}** not found. Run the pipeline first:\n\n"
        "```bash\n"
        "python -m src.generation.generate_corpus --target 100\n"
        "python -m src.preprocessing.normalize\n"
        "python -m src.exploration.llm_explorer --sample 20\n"
        "python -m src.embeddings.embed_cluster --dry-run\n"
        "```",
        icon="⚠️",
    )


# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Sans", color="#e6edf3", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ── Page 1: Taxonomy Explorer ─────────────────────────────────────────────────
def page_taxonomy() -> None:
    st.markdown('<div class="page-header">Taxonomy Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Risk themes, actions, and outcomes discovered by GPT-4o from the 200-doc sample.</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-badge">⚡ SKILL: LLM-first exploratory analysis + taxonomy mapping</div>', unsafe_allow_html=True)

    taxonomy = load_taxonomy()
    labels_df = load_llm_labels()

    if not taxonomy:
        _data_missing_warning("taxonomy.json")
        return

    # ── Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    risk_themes = taxonomy.get("risk_themes", [])
    actions = taxonomy.get("actions", [])
    outcomes = taxonomy.get("outcomes", [])
    new_labels = taxonomy.get("new_labels_proposed", [])

    col1.metric("Risk themes", len([t for t in risk_themes if t.get("retain")]))
    col2.metric("Actions", len([a for a in actions if a.get("retain")]))
    col3.metric("Outcomes", len([o for o in outcomes if o.get("retain")]))
    col4.metric("New labels proposed", len(new_labels))

    st.markdown("---")

    # ── Risk theme frequency chart
    if risk_themes:
        theme_df = pd.DataFrame(risk_themes).sort_values("count", ascending=True)
        fig = go.Figure(go.Bar(
            x=theme_df["count"],
            y=theme_df["label"].str.replace("_", " "),
            orientation="h",
            marker=dict(
                color=theme_df["count"],
                colorscale=[[0, "#8b6000"], [1, "#f0a500"]],
                showscale=False,
            ),
        ))
        fig.update_layout(
            title="Risk theme frequency",
            xaxis_title="Doc count",
            yaxis_title=None,
            height=max(300, len(theme_df) * 32),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    # ── Action frequency
    with col_a:
        if actions:
            act_df = pd.DataFrame(actions).sort_values("count", ascending=False)
            fig2 = px.bar(
                act_df, x="label", y="count",
                title="Action distribution",
                color_discrete_sequence=["#58a6ff"],
            )
            fig2.update_layout(
                xaxis_tickangle=-30,
                xaxis_title=None,
                height=300,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── New labels proposed
    with col_b:
        if new_labels:
            st.markdown("**New labels proposed by GPT-4o**")
            for nl in new_labels:
                rec = nl.get("recommendation", "")
                color = {"add_to_taxonomy": "#3fb950", "merge_into_existing": "#58a6ff"}.get(rec, "#7d8590")
                st.markdown(
                    f'<div class="memo-card">'
                    f'<span style="color:{color};font-family:IBM Plex Mono;font-size:0.75rem">{rec}</span><br>'
                    f'<strong>{nl.get("label","")}</strong> '
                    f'<span class="section-tag">{nl.get("category","")}</span>'
                    f'<br><small style="color:#7d8590">Appeared in {nl.get("count",0)} docs</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Analyst notes
    notes = taxonomy.get("analyst_notes", "")
    if notes:
        st.markdown("---")
        st.markdown("**GPT-4o analyst notes on corpus patterns**")
        st.info(notes)

    # ── Labeled docs table
    if not labels_df.empty:
        st.markdown("---")
        st.markdown("**Labeled sample documents**")
        filter_theme = st.selectbox(
            "Filter by risk theme",
            options=["All"] + sorted(labels_df["risk_theme"].dropna().unique().tolist()),
        )
        display_df = labels_df if filter_theme == "All" else labels_df[labels_df["risk_theme"] == filter_theme]
        st.dataframe(
            display_df[["issuer", "sector", "risk_theme", "action", "confidence", "ambiguous"]].reset_index(drop=True),
            use_container_width=True,
            height=300,
        )


# ── Page 2: Cluster Map ───────────────────────────────────────────────────────
def page_cluster_map() -> None:
    st.markdown('<div class="page-header">Cluster Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">UMAP 2D projection of all memo chunks. Each point is a chunk, colored by cluster assignment.</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-badge">⚡ SKILL: Rapid semantic clustering — scaling prompt prototype to embedding pipeline</div>', unsafe_allow_html=True)

    chunks_df = load_chunks()
    purity = load_cluster_purity()

    if chunks_df.empty:
        _data_missing_warning("Chunk table")
        return

    if "umap_x" not in chunks_df.columns:
        st.warning("UMAP coordinates not found. Run `embed_cluster.py` first.", icon="⚠️")
        return

    # ── Purity metrics
    if purity:
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean cluster purity", f"{purity.get('mean_cluster_purity', 0):.1%}")
        col2.metric("Adjusted Rand Index", f"{purity.get('adjusted_rand_index', 0):.4f}")
        col3.metric("Labeled docs calibrated", purity.get("n_labeled_docs", 0))
        st.markdown(
            "**Cluster purity** measures agreement between KMeans clusters and GPT-4o labels. "
            "High purity means the embedding space captures the same semantic structure as the LLM.",
        )
        st.markdown("---")

    # ── Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 2])
    with ctrl_col1:
        color_by = st.selectbox(
            "Color by",
            options=["cluster_id", "section", "doc_type", "collateral", "recommended_action"],
        )
    with ctrl_col2:
        section_filter = st.multiselect(
            "Filter by section",
            options=sorted(chunks_df["section"].dropna().unique().tolist()),
            default=[],
        )
    with ctrl_col3:
        max_points = st.slider("Max points", 500, min(10000, len(chunks_df)), 3000, 500)

    # Apply filters
    plot_df = chunks_df.dropna(subset=["umap_x", "umap_y"])
    if section_filter:
        plot_df = plot_df[plot_df["section"].isin(section_filter)]
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=42)

    if plot_df.empty:
        st.info("No points to display with current filters.")
        return

    plot_df = plot_df.copy()
    plot_df[color_by] = plot_df[color_by].astype(str).fillna("unknown")

    # ── UMAP scatter
    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color=color_by,
        hover_data=["issuer", "section", "doc_type", "net_leverage"],
        custom_data=["chunk_id", "issuer", "section", "text"],
        title=f"UMAP cluster map — {len(plot_df):,} chunks — colored by {color_by}",
        opacity=0.7,
        size_max=6,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        height=600,
        legend=dict(
            bgcolor="#1c2128",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11),
        ),
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    fig.update_yaxes(showgrid=False, zeroline=False, title=None)

    selected = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

    # ── Show selected point detail
    if selected and selected.get("selection", {}).get("points"):
        pt = selected["selection"]["points"][0]
        idx = pt.get("point_index", 0)
        if idx < len(plot_df):
            row = plot_df.iloc[idx]
            st.markdown("**Selected chunk:**")
            st.markdown(
                f'<div class="memo-card">'
                f'<strong>{row.get("issuer","")}</strong>'
                f'<span class="section-tag">{row.get("section","")}</span>'
                f'<span class="metric-pill">cluster {row.get("cluster_id","")}</span>'
                f'<p style="margin-top:0.5rem;color:#e6edf3;font-size:0.9rem">{str(row.get("text",""))[:400]}...</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Per-cluster purity table
    if purity.get("per_cluster"):
        with st.expander("Per-cluster purity breakdown"):
            purity_df = pd.DataFrame(purity["per_cluster"])
            st.dataframe(purity_df, use_container_width=True)


# ── Page 3: Similar Memo Retrieval ───────────────────────────────────────────
def page_retrieval() -> None:
    st.markdown('<div class="page-header">Similar Memo Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Semantic search over credit memos with optional structured filters.</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-badge">⚡ SKILL: LangChain VectorStoreRetriever — combined structured + semantic search</div>', unsafe_allow_html=True)

    # ── Query input
    query_text = st.text_area(
        "Query",
        placeholder="e.g. covenant headroom deteriorating, sponsor unlikely to support refinancing",
        height=80,
    )

    # ── Filters
    with st.expander("Structured filters (optional)", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            min_leverage = st.number_input("Min net leverage (x)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
            max_leverage = st.number_input("Max net leverage (x)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
        with fc2:
            collateral_filter = st.selectbox(
                "Collateral",
                options=["Any", "first_lien", "second_lien", "unsecured", "first_lien_second_lien_split"],
            )
        with fc3:
            section_filter = st.selectbox(
                "Section",
                options=["Any", "executive_summary_reco", "merits_and_concerns",
                         "transaction_overview", "rep_risk_esg", "company_description"],
            )
        with fc4:
            top_k = st.slider("Results", min_value=3, max_value=20, value=8)

    # ── Build where clause
    def _build_where() -> dict | None:
        clauses = []
        if min_leverage > 0:
            clauses.append({"net_leverage": {"$gte": float(min_leverage)}})
        if max_leverage > 0 and max_leverage >= min_leverage:
            clauses.append({"net_leverage": {"$lte": float(max_leverage)}})
        if collateral_filter != "Any":
            clauses.append({"collateral": collateral_filter})
        if not clauses:
            return None
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]

    sections = None if section_filter == "Any" else [section_filter]

    if st.button("Search", use_container_width=False):
        if not query_text.strip():
            st.warning("Enter a query to search.", icon="⚠️")
            return

        try:
            from src.retrieval.retriever import SimilarCaseRetriever
            retriever = SimilarCaseRetriever(top_k=top_k)
        except Exception as exc:
            st.error(f"Could not load retriever: {exc}")
            return

        with st.spinner("Searching..."):
            try:
                results = retriever.query(
                    text=query_text,
                    filters=_build_where(),
                    top_k=top_k,
                    sections=sections,
                )
            except Exception as exc:
                st.error(f"Search failed: {exc}. Run embed_cluster.py first.")
                return

        if not results:
            st.info("No results above similarity threshold. Try a different query or lower the threshold in config.yaml.")
            return

        st.markdown(f"**{len(results)} results** for: *{query_text[:80]}*")
        st.markdown("---")

        for r in results:
            lev = f"{r.net_leverage:.1f}x" if r.net_leverage else "—"
            coll = r.metadata.get("collateral", "").replace("_", " ") or "—"
            sector = r.metadata.get("sector", "") or "—"
            action = r.metadata.get("recommended_action", "").replace("_", " ") or "—"

            st.markdown(
                f'<div class="memo-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<strong style="font-size:1rem">{r.issuer}</strong>'
                f'<span class="score-badge">sim {r.score:.3f}</span>'
                f'</div>'
                f'<span class="section-tag">{r.section.replace("_"," ")}</span>'
                f'<div style="margin-top:0.6rem">'
                f'<span class="metric-pill">leverage {lev}</span>'
                f'<span class="metric-pill">{coll}</span>'
                f'<span class="metric-pill">{sector[:20]}</span>'
                f'<span class="metric-pill">→ {action}</span>'
                f'</div>'
                f'<p style="margin-top:0.6rem;color:#c9d1d9;font-size:0.88rem;line-height:1.6">'
                f'{r.text[:350]}{"..." if len(r.text) > 350 else ""}'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Taxonomy-filtered search
    st.markdown("---")
    st.markdown("**Search by taxonomy label**")
    tc1, tc2, tc3 = st.columns([3, 3, 2])
    with tc1:
        taxonomy = load_taxonomy()
        theme_options = [t["label"] for t in taxonomy.get("risk_themes", []) if t.get("retain")]
        selected_theme = st.selectbox("Risk theme", options=[""] + theme_options)
    with tc2:
        theme_query = st.text_input("Refine with query (optional)", placeholder="e.g. liquidity pressure")
    with tc3:
        st.markdown("<br>", unsafe_allow_html=True)
        theme_search = st.button("Search by theme")

    if theme_search and selected_theme:
        try:
            from src.retrieval.retriever import TaxonomyRetriever
            tr = TaxonomyRetriever(top_k=6)
            results = tr.query_by_theme(
                risk_theme=selected_theme,
                query_text=theme_query or None,
            )
            for r in results:
                st.markdown(
                    f'<div class="memo-card">'
                    f'<strong>{r.issuer}</strong>'
                    f'<span class="score-badge" style="float:right">sim {r.score:.3f}</span>'
                    f'<p style="color:#c9d1d9;font-size:0.88rem;margin-top:0.5rem">{r.text[:280]}...</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.error(f"Taxonomy search failed: {exc}")


# ── Page 4: Functional Decomposition ─────────────────────────────────────────
def page_decomposition() -> None:
    st.markdown('<div class="page-header">Functional Decomposition View</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Auto-extracted analytical slots alongside source memo text.</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-badge">⚡ SKILL: Functional decomposition automation — trigger · risk signal · analyst stance · forward view</div>', unsafe_allow_html=True)

    labels_df = load_llm_labels()

    if labels_df.empty:
        _data_missing_warning("llm_labels.parquet")
        return

    # ── Doc selector
    issuers = sorted(labels_df["issuer"].dropna().unique().tolist())
    selected_issuer = st.selectbox("Select issuer", options=issuers)

    doc_rows = labels_df[labels_df["issuer"] == selected_issuer]
    if doc_rows.empty:
        st.info("No labeled docs for this issuer.")
        return

    row = doc_rows.iloc[0]

    # ── Memo metadata strip
    st.markdown(
        f'<div style="background:#1c2128;border:1px solid #30363d;border-radius:6px;padding:0.75rem 1rem;margin-bottom:1rem">'
        f'<span class="metric-pill">{row.get("sector","")}</span>'
        f'<span class="metric-pill">{row.get("rating_bucket","")}</span>'
        f'<span class="metric-pill">{row.get("doc_type","").replace("_"," ")}</span>'
        + (f'<span class="metric-pill">leverage {row.get("net_leverage"):.1f}x</span>' if row.get("net_leverage") else "")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Confidence and ambiguity
    conf = float(row.get("confidence", 0))
    amb = bool(row.get("ambiguous", False))
    conf_color = "#3fb950" if conf >= 0.75 else "#f0a500" if conf >= 0.5 else "#f85149"

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk theme", str(row.get("risk_theme", "—")).replace("_", " "))
    c2.metric("Confidence", f"{conf:.0%}")
    c3.metric("Ambiguous", "Yes" if amb else "No")

    st.markdown("---")

    # ── Four decomposition slots
    st.markdown("**Auto-extracted decomposition slots**")
    slot_cols = st.columns(4)
    slots = [
        ("TRIGGER", "trigger", "What prompted this memo?"),
        ("RISK SIGNAL", "risk_signal", "Specific credit concern"),
        ("ANALYST STANCE", "analyst_stance", "Recommendation and rationale"),
        ("FORWARD VIEW", "forward_view", "Expected outcome"),
    ]
    for col, (label, field, hint) in zip(slot_cols, slots):
        with col:
            text = str(row.get(field, "—"))
            st.markdown(
                f'<div class="slot-card">'
                f'<div class="slot-label">{label}</div>'
                f'<div style="color:#7d8590;font-size:0.7rem;margin-bottom:0.5rem">{hint}</div>'
                f'<div class="slot-text">{text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Action and outcome
    act_col, out_col = st.columns(2)
    with act_col:
        action = str(row.get("action", "—")).replace("_", " ")
        st.markdown(f"**Recommended action:** `{action}`")
    with out_col:
        outcome = str(row.get("outcome", "—")).replace("_", " ")
        st.markdown(f"**Expected outcome:** `{outcome}`")

    # ── Browse all labeled docs
    st.markdown("---")
    with st.expander("Browse all labeled documents"):
        display_cols = ["issuer", "sector", "risk_theme", "action", "confidence", "ambiguous"]
        available = [c for c in display_cols if c in labels_df.columns]
        st.dataframe(
            labels_df[available].sort_values("confidence", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=350,
        )


# ── Sidebar navigation ────────────────────────────────────────────────────────
def sidebar() -> str:
    with st.sidebar:
        st.markdown("## Credit Memo\nIntelligence Platform")
        st.markdown("---")
        st.markdown("### Navigation")

        page = st.radio(
            label="page",
            options=[
                "📊  Taxonomy Explorer",
                "🗺️  Cluster Map",
                "🔍  Similar Memo Search",
                "🧩  Decomposition View",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Pipeline status")

        checks = {
            "Corpus": ROOT / "data" / "raw" / "corpus.jsonl",
            "Chunks (DuckDB)": ROOT / "data" / "processed" / "corpus.duckdb",
            "Taxonomy": ROOT / "data" / "processed" / "taxonomy.json",
            "LLM labels": ROOT / "data" / "processed" / "llm_labels.parquet",
            "Embeddings": ROOT / "data" / "processed" / "embeddings.npy",
            "ChromaDB": ROOT / "data" / "chroma_db",
        }
        for label, path in checks.items():
            exists = path.exists()
            icon = "🟢" if exists else "🔴"
            st.markdown(f"{icon} {label}")

        st.markdown("---")
        st.markdown(
            '<div style="font-family:IBM Plex Mono;font-size:0.65rem;color:#7d8590">'
            'MVP · Phase 1–6 complete<br>'
            'BGE-small · KMeans · ChromaDB<br>'
            'LangChain · Azure OpenAI / OpenAI'
            '</div>',
            unsafe_allow_html=True,
        )

    return page


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    page = sidebar()

    if "Taxonomy" in page:
        page_taxonomy()
    elif "Cluster" in page:
        page_cluster_map()
    elif "Search" in page:
        page_retrieval()
    elif "Decomposition" in page:
        page_decomposition()


if __name__ == "__main__":
    main()
