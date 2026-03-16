"""
Call Center AI Quality Monitor
Streamlit Dashboard — Main Entry Point

Run with: streamlit run app.py
"""

import os
import sys
import json
import time
import tempfile
import warnings
from pathlib import Path

# ── Path fix: runs BEFORE any imports, covers all streamlit run modes ───────────
_HERE = Path(os.path.abspath(__file__)).parent
for _p in [str(_HERE), os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration  ← MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CallCenter AI Monitor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports (after set_page_config) ──────────────────────────────────────
try:
    from analyzer import CallCenterAnalyzer, InteractionAnalysis
    from speech_to_text import SpeechToTextConverter
    from report_generator import (
        generate_csv_bytes, generate_json_bytes, generate_text_report
    )
except ModuleNotFoundError as e:
    st.error(f"❌ Import error: **{e}**")
    st.code(
        "# Run from INSIDE the project folder:\n"
        "cd callcenter_v3\n"
        "streamlit run app.py",
        language="bash",
    )
    st.info("Python searched in:\n\n" + "\n".join(f"- `{p}`" for p in sys.path[:8]))
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark Professional Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --bg-primary:    #f4f6f9;
    --bg-secondary:  #eaecf2;
    --bg-card:       #ffffff;
    --bg-card-hover: #f0f4ff;
    --accent-blue:   #1d4ed8;
    --accent-cyan:   #0369a1;
    --accent-green:  #15803d;
    --accent-red:    #b91c1c;
    --accent-amber:  #b45309;
    --accent-purple: #6d28d9;
    --text-primary:  #0f172a;
    --text-secondary:#1e293b;
    --text-muted:    #475569;
    --border:        #cbd5e1;
    --border-accent: #94a3b8;
    --font-main: 'Space Grotesk', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }

  html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
  }

  .stApp {
    background: linear-gradient(135deg, #f4f6f9 0%, #eef2fb 50%, #f1f5f9 100%) !important;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border);
    box-shadow: 2px 0 8px rgba(0,0,0,0.06);
  }
  section[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary) !important; }
  section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  /* ── Metric Cards ── */
  [data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    transition: all 0.2s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
  }
  [data-testid="metric-container"]:hover { background: var(--bg-card-hover) !important; }
  [data-testid="metric-container"] label { color: var(--text-muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
  [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }
  [data-testid="stMetricDelta"] { color: var(--text-muted) !important; font-size: 0.78rem !important; }
  [data-testid="stMetricDelta"] svg { display: none; }

  /* ── KPI metric value: shrink font so text never truncates ── */
  [data-testid="stMetricValue"] > div {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }
  [data-testid="metric-container"] label {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-accent) !important;
    border-radius: 12px !important;
    padding: 12px;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover { border-color: var(--accent-blue) !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #0369a1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-main) !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(29,78,216,0.25);
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(29,78,216,0.35) !important;
  }

  /* ── Download buttons ── */
  .stDownloadButton > button {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
  }
  .stDownloadButton > button:hover {
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    background: #eff6ff !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px;
    border: 1px solid var(--border);
  }
  .stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: #ffffff !important;
  }

  /* ── Dataframe ── */
  .stDataFrame { background: var(--bg-card) !important; border-radius: 12px; border: 1px solid var(--border); }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
  }
  .streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
  }

  /* ── Custom cards ── */
  .ai-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.2s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .ai-card:hover { border-color: var(--accent-blue); box-shadow: 0 3px 10px rgba(29,78,216,0.1); }

  .section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent-blue);
    margin-bottom: 12px;
  }

  .sentiment-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .badge-positive { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
  .badge-neutral  { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
  .badge-negative { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

  .score-bar-outer {
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
    margin: 6px 0;
  }
  .score-bar-inner {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .transcript-bubble-agent {
    background: #eff6ff;
    border-left: 3px solid #1d4ed8;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
  }
  .transcript-bubble-customer {
    background: #f0f9ff;
    border-left: 3px solid #0369a1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
  }
  .speaker-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 3px;
  }
  .agent-label { color: #1d4ed8; }
  .customer-label { color: #0369a1; }

  /* ── Header ── */
  .main-header {
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(29,78,216,0.05) 0%, transparent 70%);
    pointer-events: none;
  }

  /* ── Info/warning banners ── */
  .info-banner {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px 18px;
    color: #1e40af;
    font-size: 0.9rem;
    margin: 12px 0;
  }
  .success-banner {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 14px 18px;
    color: #15803d;
    font-size: 0.9rem;
  }
  .warning-banner {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 14px 18px;
    color: #92400e;
    font-size: 0.9rem;
  }

  /* ── Inputs & selects ── */
  .stSelectbox > div, .stRadio > div { color: var(--text-primary) !important; }
  .stTextInput input, .stSelectbox select { background: #ffffff !important; color: var(--text-primary) !important; border: 1px solid var(--border) !important; }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-secondary); }
  ::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "analyzer": None,
        "stt": None,
        "analysis_result": None,
        "transcript_result": None,
        "history": [],           # list of past analyses
        "models_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all models once and cache them."""
    analyzer = CallCenterAnalyzer()
    stt = SpeechToTextConverter(model_size="base")
    return analyzer, stt


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_COLOR = {
    "positive": "#10b981",
    "neutral": "#94a3b8",
    "negative": "#ef4444",
}

EMOTION_COLORS = [
    "#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b",
    "#10b981", "#ef4444", "#ec4899", "#6366f1",
]

def sentiment_badge_html(label: str) -> str:
    cls = f"badge-{label.lower()}"
    return f'<span class="sentiment-badge {cls}">{label.upper()}</span>'


def score_bar_html(score: float, color: str) -> str:
    pct = int(score * 100)
    return (
        f"<div class='score-bar-outer'>"
        f"<div class='score-bar-inner' style='width:{pct}%;background:{color};'></div>"
        f"</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0;'>
      <div style='font-size:2rem'>🎙️</div>
      <div style='font-size:1.1rem; font-weight:700; color:#0f172a;'>CallCenter AI</div>
      <div style='font-size:0.72rem; color:#475569; letter-spacing:0.1em; text-transform:uppercase;'>Quality Monitor v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-title">📂 Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Choose input type",
        ["Upload File", "Use Sample Transcript"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<div class="section-title">⚙️ Analysis Settings</div>', unsafe_allow_html=True)

    whisper_model = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower. 'base' is recommended.",
    )

    run_emotion = st.toggle("Enable Emotion Detection", value=True)
    run_agent = st.toggle("Enable Agent Analysis", value=True)

    st.divider()

    st.markdown('<div class="section-title">📊 History</div>', unsafe_allow_html=True)
    if st.session_state.history:
        st.caption(f"{len(st.session_state.history)} interaction(s) analyzed")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No analyses yet")

    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem; color:#334155; text-align:center;'>
      Models: RoBERTa · BERT · DistilBERT<br>
      STT: OpenAI Whisper<br>
      <span style='color:#1d4ed8;'>Built with Python + Streamlit</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class='main-header'>
  <div style='display:flex; align-items:center; gap:16px;'>
    <div style='font-size:2.5rem;'>🎙️</div>
    <div>
      <h1 style='margin:0; font-size:1.8rem; font-weight:700; color:#0f172a;'>
        Call Center AI Quality Monitor
      </h1>
      <p style='margin:4px 0 0; color:#64748b; font-size:0.9rem;'>
        Automated sentiment analysis &amp; agent assessment powered by RoBERTa · BERT · DistilBERT
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading Section
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.models_loaded:
    with st.spinner("🔄 Loading AI models (RoBERTa, BERT, DistilBERT, Whisper)..."):
        try:
            analyzer, stt = load_models()
            st.session_state.analyzer = analyzer
            st.session_state.stt = stt
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"Model loading error: {e}")
            st.stop()

    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-title">📥 Input & Analysis</div>', unsafe_allow_html=True)

input_col, _ = st.columns([2, 1])

with input_col:
    uploaded_file = None
    sample_choice = None

    if input_mode == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload an audio or transcript file",
            type=["wav", "mp3", "m4a", "ogg", "flac", "txt", "csv"],
            help="Audio: WAV, MP3, M4A, OGG, FLAC | Transcript: TXT (Agent:/Customer: format) or CSV (speaker,text columns)",
        )
        if uploaded_file:
            ftype = uploaded_file.name.split(".")[-1].lower()
            is_audio = ftype in {"wav", "mp3", "m4a", "ogg", "flac"}
            st.markdown(f"""
            <div class='info-banner'>
              {'🎵' if is_audio else '📄'} <strong>{uploaded_file.name}</strong>
              &nbsp;·&nbsp; {ftype.upper()} &nbsp;·&nbsp; {uploaded_file.size / 1024:.1f} KB
              {'&nbsp;·&nbsp; Will be transcribed with Whisper' if is_audio else ''}
            </div>
            """, unsafe_allow_html=True)

    else:
        sample_files = {
            "📞 Tech Support – Internet Outage (TXT)": str(ROOT / "sample_data" / "sample_transcript_1.txt"),
            "🏦 Banking – Billing Dispute (TXT)": str(ROOT / "sample_data" / "sample_transcript_2.txt"),
            "💊 Pharmacy – Prescription Inquiry (CSV)": str(ROOT / "sample_data" / "sample_transcript_3.csv"),
        }
        sample_choice = st.selectbox("Choose a sample interaction:", list(sample_files.keys()))
        st.markdown("""
        <div class='info-banner'>
          💡 These are pre-labeled transcripts demonstrating different call types and customer sentiment arcs.
        </div>
        """, unsafe_allow_html=True)

    analyze_btn = st.button("🚀 Analyze Interaction", use_container_width=False)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Pipeline
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn:
    analyzer = st.session_state.analyzer
    stt = st.session_state.stt

    transcript_result = None
    error_msg = None

    progress_bar = st.progress(0, text="Initializing...")

    try:
        # ── Step 1: Load / transcribe input ──────────────────────────────────
        progress_bar.progress(10, text="📂 Loading input...")

        if input_mode == "Upload File" and uploaded_file is not None:
            ftype = uploaded_file.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(suffix=f".{ftype}", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            if ftype in {"wav", "mp3", "m4a", "ogg", "flac"}:
                progress_bar.progress(25, text="🎙️ Transcribing audio with Whisper...")
                transcript_result = stt.convert_audio_to_text(tmp_path)
            else:
                progress_bar.progress(25, text="📄 Parsing transcript file...")
                transcript_result = stt.parse_transcript_file(tmp_path)

            os.unlink(tmp_path)

        elif input_mode == "Use Sample Transcript":
            path = sample_files[sample_choice]
            progress_bar.progress(25, text="📄 Loading sample transcript...")
            transcript_result = stt.parse_transcript_file(path)

        else:
            error_msg = "Please upload a file or select a sample transcript."

        # ── Step 2: NLP Analysis ──────────────────────────────────────────────
        if transcript_result and not error_msg:
            progress_bar.progress(45, text="🤖 Running RoBERTa sentiment analysis...")
            time.sleep(0.3)

            progress_bar.progress(60, text="🤖 Running BERT analysis...")
            time.sleep(0.3)

            progress_bar.progress(72, text="🤖 Running DistilBERT analysis...")
            time.sleep(0.3)

            progress_bar.progress(82, text="😊 Detecting emotions...")

            analysis = analyzer.analyze_interaction(transcript_result)

            progress_bar.progress(92, text="📊 Generating insights...")
            time.sleep(0.2)

            # Save to history
            history_entry = {
                "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                "source": uploaded_file.name if uploaded_file else sample_choice,
                "sentiment": analysis.customer_overall_sentiment,
                "agent_score": analysis.agent_behavior.overall_score,
                "resolved": analysis.issue_resolved,
            }
            st.session_state.history.append(history_entry)

            st.session_state.analysis_result = analysis
            st.session_state.transcript_result = transcript_result

            progress_bar.progress(100, text="✅ Analysis complete!")
            time.sleep(0.5)
            progress_bar.empty()

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Analysis failed: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())

    if error_msg:
        progress_bar.empty()
        st.warning(error_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.analysis_result is not None:
    analysis: InteractionAnalysis = st.session_state.analysis_result
    transcript: dict = st.session_state.transcript_result

    st.divider()
    st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # KPI Row
    # ──────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(analysis.customer_overall_sentiment, "❓")
        st.metric(
            "Sentiment",
            f"{analysis.customer_overall_sentiment.capitalize()} {emoji}",
            f"{analysis.customer_sentiment_score:.0%}",
        )
    with k2:
        # Truncate long emotion names to avoid wrapping
        emotion = analysis.customer_dominant_emotion
        emotion_short = emotion if len(emotion) <= 14 else emotion[:13] + "…"
        st.metric("Top Emotion", emotion_short)
    with k3:
        score_pct = f"{analysis.agent_behavior.overall_score:.0%}"
        delta = "Excellent" if analysis.agent_behavior.overall_score >= 0.75 else (
            "Good" if analysis.agent_behavior.overall_score >= 0.55 else "Needs work"
        )
        st.metric("Agent Score", score_pct, delta)
    with k4:
        resolved_label = "✅ Resolved" if analysis.issue_resolved else "❌ Unresolved"
        st.metric("Resolution", resolved_label, analysis.sentiment_after_resolution.capitalize())

    st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # Tabs
    # ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Sentiment Trends",
        "😊 Emotion Analysis",
        "🎧 Agent Performance",
        "📝 Transcript",
        "📋 Summary & Export",
    ])


    # ── TAB 1: Sentiment Trends ───────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([3, 1])

        with col_a:
            st.markdown('<div class="section-title">Sentiment Over Conversation</div>', unsafe_allow_html=True)

            # Build timeline data
            rows = []
            for seg in analysis.segment_analyses:
                score = seg.sentiment.score
                label = seg.sentiment.label
                # Map to -1 / 0 / +1 scale
                numeric = {"positive": score, "neutral": 0.0, "negative": -score}[label]
                rows.append({
                    "Time (s)": round((seg.start + seg.end) / 2, 1),
                    "Score": numeric,
                    "Label": label.capitalize(),
                    "Speaker": seg.speaker,
                    "Text": seg.text[:60] + ("..." if len(seg.text) > 60 else ""),
                })

            df_sent = pd.DataFrame(rows)

            fig_line = go.Figure()

            # Shaded regions
            fig_line.add_hrect(y0=0, y1=1.05, fillcolor="rgba(16,185,129,0.06)", line_width=0)
            fig_line.add_hrect(y0=-1.05, y1=0, fillcolor="rgba(239,68,68,0.06)", line_width=0)
            fig_line.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)

            for speaker, color, dash in [("Customer", "#06b6d4", "solid"), ("Agent", "#3b82f6", "dash")]:
                df_sp = df_sent[df_sent["Speaker"] == speaker]
                if df_sp.empty:
                    continue
                fig_line.add_trace(go.Scatter(
                    x=df_sp["Time (s)"],
                    y=df_sp["Score"],
                    mode="lines+markers",
                    name=speaker,
                    line=dict(color=color, width=2.5, dash=dash),
                    marker=dict(size=8, color=color, symbol="circle",
                                line=dict(color="rgba(0,0,0,0.3)", width=1)),
                    hovertemplate="<b>%{customdata}</b><br>Time: %{x}s<br>Sentiment: %{y:.2f}<extra></extra>",
                    customdata=df_sp["Text"],
                ))

            fig_line.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0.9)",
                font=dict(family="Space Grotesk", color="#334155"),
                legend=dict(
                    bgcolor="rgba(255,255,255,0)", font=dict(color="#334155"),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                ),
                xaxis=dict(
                    title="Time (seconds)", gridcolor="#e2e8f0",
                    color="#475569", showline=True, linecolor="#cbd5e1",
                ),
                yaxis=dict(
                    title="Sentiment Score", range=[-1.1, 1.1],
                    gridcolor="#e2e8f0", color="#475569",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
                ),
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig_line, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

            model_data = {
                "RoBERTa": (analysis.roberta_result.label, analysis.roberta_result.score, "#3b82f6"),
                "BERT": (analysis.bert_result.label, analysis.bert_result.score, "#8b5cf6"),
                "DistilBERT": (analysis.distilbert_result.label, analysis.distilbert_result.score, "#06b6d4"),
            }

            for model_name, (label, score, color) in model_data.items():
                badge = sentiment_badge_html(label)
                bar = score_bar_html(score, color)
                st.markdown(
                    f"<div class='ai-card' style='padding:14px;'>"
                    f"<div style='font-size:0.8rem;font-weight:600;color:#334155;margin-bottom:6px;'>{model_name}</div>"
                    f"{badge}"
                    f"{bar}"
                    f"<div style='font-size:0.72rem;color:#64748b;'>{score:.0%} confidence</div>"
                    f"</div>",
                    unsafe_allow_html=True)

        # Sentiment distribution bar chart
        st.markdown('<div class="section-title">Sentiment Distribution by Speaker</div>', unsafe_allow_html=True)

        dist_rows = []
        for seg in analysis.segment_analyses:
            dist_rows.append({"Speaker": seg.speaker, "Sentiment": seg.sentiment.label.capitalize()})
        df_dist = pd.DataFrame(dist_rows)

        if not df_dist.empty:
            dist_counts = df_dist.groupby(["Speaker", "Sentiment"]).size().reset_index(name="Count")
            fig_bar = px.bar(
                dist_counts, x="Speaker", y="Count", color="Sentiment",
                color_discrete_map={"Positive": "#10b981", "Neutral": "#94a3b8", "Negative": "#ef4444"},
                barmode="group",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0.9)",
                font=dict(family="Space Grotesk", color="#334155"),
                legend=dict(bgcolor="rgba(255,255,255,0)"),
                xaxis=dict(gridcolor="#e2e8f0", color="#475569"),
                yaxis=dict(gridcolor="#e2e8f0", color="#475569"),
                height=260, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


    # ── TAB 2: Emotion Analysis ───────────────────────────────────────────────
    with tab2:
        ec1, ec2 = st.columns([1, 1])

        with ec1:
            st.markdown('<div class="section-title">Customer Emotion Distribution</div>', unsafe_allow_html=True)

            emotions = analysis.customer_emotion_distribution
            if emotions:
                fig_pie = go.Figure(go.Pie(
                    labels=list(emotions.keys()),
                    values=list(emotions.values()),
                    hole=0.52,
                    marker=dict(colors=EMOTION_COLORS[:len(emotions)],
                                line=dict(color="#0a0e1a", width=2)),
                    textinfo="label+percent",
                    textfont=dict(family="Space Grotesk", size=11),
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(255,255,255,0)",
                    font=dict(family="Space Grotesk", color="#334155"),
                    legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(size=11)),
                    showlegend=True,
                    height=360,
                    margin=dict(l=10, r=10, t=30, b=10),
                    annotations=[dict(
                        text=f"<b>{analysis.customer_dominant_emotion}</b>",
                        x=0.5, y=0.5, font_size=13, font_color="#0f172a",
                        showarrow=False,
                    )],
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with ec2:
            st.markdown('<div class="section-title">Emotion Intensity Over Time</div>', unsafe_allow_html=True)

            emo_rows = []
            for seg in analysis.segment_analyses:
                if seg.emotion and seg.speaker.lower() == "customer":
                    emo_rows.append({
                        "Time": (seg.start + seg.end) / 2,
                        "Emotion": seg.emotion.emotion,
                        "Intensity": seg.emotion.score,
                    })

            if emo_rows:
                df_emo = pd.DataFrame(emo_rows)
                fig_emo = px.scatter(
                    df_emo, x="Time", y="Emotion", size="Intensity",
                    color="Emotion", size_max=30,
                    color_discrete_sequence=EMOTION_COLORS,
                )
                fig_emo.update_layout(
                    paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0.9)",
                    font=dict(family="Space Grotesk", color="#334155"),
                    showlegend=False,
                    height=360,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title="Time (s)", gridcolor="#e2e8f0", color="#475569"),
                    yaxis=dict(gridcolor="#e2e8f0", color="#475569"),
                )
                st.plotly_chart(fig_emo, use_container_width=True)
            else:
                st.info("No customer emotion timeline data available.")

        # Emotion heatmap
        st.markdown('<div class="section-title">Emotion Heatmap</div>', unsafe_allow_html=True)

        if emotions:
            sorted_emos = dict(sorted(emotions.items(), key=lambda x: -x[1]))
            for emo, score in sorted_emos.items():
                idx = list(sorted_emos.keys()).index(emo)
                color = EMOTION_COLORS[idx % len(EMOTION_COLORS)]
                bar = score_bar_html(score, color)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:12px;margin:4px 0;'>"
                    f"<div style='width:140px;font-size:0.82rem;color:#334155;'>{emo}</div>"
                    f"<div style='flex:1;'>{bar}</div>"
                    f"<div style='width:50px;font-size:0.8rem;color:#64748b;text-align:right;'>{score:.0%}</div>"
                    f"</div>",
                    unsafe_allow_html=True)


    # ── TAB 3: Agent Performance ──────────────────────────────────────────────
    with tab3:
        ag1, ag2 = st.columns([1, 1])

        with ag1:
            st.markdown('<div class="section-title">Performance Scores</div>', unsafe_allow_html=True)

            behavior = analysis.agent_behavior
            metrics = [
                ("Empathy", behavior.empathy_score, "#10b981"),
                ("Professionalism", behavior.professionalism_score, "#3b82f6"),
                ("Problem-Solving", behavior.problem_solving_score, "#8b5cf6"),
                ("Overall Score", behavior.overall_score, "#f59e0b"),
            ]

            fig_radar = go.Figure(go.Scatterpolar(
                r=[behavior.empathy_score, behavior.professionalism_score,
                   behavior.problem_solving_score, behavior.overall_score,
                   behavior.empathy_score],  # close
                theta=["Empathy", "Professionalism", "Problem-Solving", "Overall", "Empathy"],
                fill="toself",
                fillcolor="rgba(59,130,246,0.15)",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=6, color="#3b82f6"),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(255,255,255,0.9)",
                    radialaxis=dict(
                        visible=True, range=[0, 1], tickfont=dict(size=9, color="#475569"),
                        gridcolor="#e2e8f0", linecolor="rgba(30,41,59,0.8)",
                    ),
                    angularaxis=dict(
                        tickfont=dict(family="Space Grotesk", size=11, color="#334155"),
                        gridcolor="#e2e8f0", linecolor="rgba(30,41,59,0.8)",
                    ),
                ),
                paper_bgcolor="rgba(255,255,255,0)",
                font=dict(family="Space Grotesk", color="#334155"),
                showlegend=False,
                height=340,
                margin=dict(l=30, r=30, t=30, b=30),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with ag2:
            st.markdown('<div class="section-title">Behavior Indicators</div>', unsafe_allow_html=True)

            for m_name, m_score, m_color in metrics:
                bar = score_bar_html(m_score, m_color)
                st.markdown(
                    f"<div class='ai-card' style='padding:14px 18px;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='font-size:0.85rem;font-weight:600;color:#1e293b;'>{m_name}</span>"
                    f"<span style='font-size:1rem;font-weight:700;color:{m_color};'>{m_score:.0%}</span>"
                    f"</div>"
                    f"{bar}"
                    f"</div>",
                    unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:8px;">Behavior Flags</div>', unsafe_allow_html=True)

        flag_cols = st.columns(2)
        for i, flag in enumerate(behavior.flags):
            with flag_cols[i % 2]:
                bg = "rgba(16,185,129,0.08)" if "✅" in flag else (
                    "rgba(245,158,11,0.08)" if "⚠️" in flag else "rgba(148,163,184,0.05)"
                )
                bd = "#10b981" if "✅" in flag else ("#f59e0b" if "⚠️" in flag else "#475569")
                st.markdown(f"""
                <div style='background:{bg}; border:1px solid {bd}40; border-radius:8px;
                            padding:10px 14px; margin:4px 0; font-size:0.85rem; color:#1e293b;'>
                  {flag}
                </div>
                """, unsafe_allow_html=True)

        # Resolution gauge
        st.markdown('<div class="section-title" style="margin-top:16px;">Resolution Assessment</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            resolved_color = "#10b981" if analysis.issue_resolved else "#ef4444"
            resolved_text = "Issue Resolved" if analysis.issue_resolved else "Issue Unresolved"
            st.markdown(f"""
            <div class='ai-card' style='text-align:center; padding:24px;'>
              <div style='font-size:2.5rem;'>{'✅' if analysis.issue_resolved else '❌'}</div>
              <div style='font-size:1rem; font-weight:700; color:{resolved_color}; margin-top:8px;'>{resolved_text}</div>
              <div style='font-size:0.8rem; color:#64748b; margin-top:4px;'>
                Post-call sentiment: <strong>{analysis.sentiment_after_resolution.capitalize()}</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=analysis.resolution_score * 100,
                title=dict(text="Resolution Confidence", font=dict(color="#334155", size=13)),
                delta=dict(reference=70, increasing=dict(color="#10b981"), decreasing=dict(color="#ef4444")),
                number=dict(suffix="%", font=dict(color="#0f172a", size=28)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#475569", tickfont=dict(color="#475569")),
                    bar=dict(color="#1d4ed8"),
                    bgcolor="rgba(255,255,255,0.9)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 40], color="rgba(239,68,68,0.15)"),
                        dict(range=[40, 65], color="rgba(245,158,11,0.1)"),
                        dict(range=[65, 100], color="rgba(16,185,129,0.1)"),
                    ],
                    threshold=dict(line=dict(color="#f59e0b", width=2), thickness=0.75, value=65),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                font=dict(family="Space Grotesk", color="#334155"),
                height=220, margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


    # ── TAB 4: Transcript ─────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Annotated Conversation Transcript</div>', unsafe_allow_html=True)

        lang = transcript.get("language", "en")
        src = transcript.get("source", "unknown")
        dur = transcript.get("duration", 0)
        st.caption(f"Source: {src.replace('_', ' ').title()} · Language: {lang.upper()} · Duration: {dur:.1f}s")

        for seg in analysis.segment_analyses:
            is_agent = seg.speaker.lower() == "agent"
            bubble_class = "transcript-bubble-agent" if is_agent else "transcript-bubble-customer"
            label_class = "agent-label" if is_agent else "customer-label"
            icon = "🎧" if is_agent else "👤"

            sent_color = SENTIMENT_COLOR.get(seg.sentiment.label, "#94a3b8")
            emo_str = ""
            if seg.emotion:
                emo_str = f'<span style="font-size:0.7rem; color:#64748b; margin-left:8px;">· {seg.emotion.emotion}</span>'

            st.markdown(f"""
            <div class='{bubble_class}'>
              <div class='speaker-label {label_class}'>{icon} {seg.speaker}
                <span style='font-size:0.65rem; color:#334155; font-weight:400; margin-left:8px;'>
                  {seg.start:.1f}s – {seg.end:.1f}s
                </span>
              </div>
              <div style='font-size:0.9rem; color:#1e293b; line-height:1.5;'>{seg.text}</div>
              <div style='margin-top:4px;'>
                <span style='font-size:0.7rem; color:{sent_color}; font-weight:600;'>
                  ● {seg.sentiment.label.capitalize()} ({seg.sentiment.score:.0%})
                </span>
                {emo_str}
              </div>
            </div>
            """, unsafe_allow_html=True)


    # ── TAB 5: Summary & Export ───────────────────────────────────────────────
    with tab5:
        s1, s2 = st.columns([3, 2])

        with s1:
            st.markdown('<div class="section-title">Interaction Summary</div>', unsafe_allow_html=True)

            sentiment_icon = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(
                analysis.customer_overall_sentiment, "⚪"
            )
            resolution_icon = "✅" if analysis.issue_resolved else "❌"

            summary_items = [
                ("Customer Sentiment", f"{sentiment_icon} {analysis.customer_overall_sentiment.capitalize()} ({analysis.customer_sentiment_score:.0%})"),
                ("Dominant Emotion", f"😊 {analysis.customer_dominant_emotion}"),
                ("Agent Overall Score", f"🎯 {analysis.agent_behavior.overall_score:.0%}"),
                ("Empathy", f"❤️ {analysis.agent_behavior.empathy_score:.0%}"),
                ("Professionalism", f"💼 {analysis.agent_behavior.professionalism_score:.0%}"),
                ("Problem-Solving", f"🔧 {analysis.agent_behavior.problem_solving_score:.0%}"),
                ("Issue Resolution", f"{resolution_icon} {'Resolved' if analysis.issue_resolved else 'Unresolved'}"),
                ("Post-Call Sentiment", f"📊 {analysis.sentiment_after_resolution.capitalize()}"),
                ("Total Turns", f"💬 {analysis.total_turns} ({analysis.customer_turns} customer, {analysis.agent_turns} agent)"),
                ("Processing Time", f"⚡ {analysis.processing_time:.2f}s"),
            ]

            table_rows = "".join([
                f"<tr><td style='padding:8px 12px; color:#475569; font-size:0.82rem;'>{k}</td>"
                f"<td style='padding:8px 12px; color:#1e293b; font-size:0.85rem; font-weight:500;'>{v}</td></tr>"
                for k, v in summary_items
            ])

            st.markdown(f"""
            <div class='ai-card'>
              <table style='width:100%; border-collapse:collapse;'>
                {table_rows}
              </table>
            </div>
            """, unsafe_allow_html=True)

        with s2:
            st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)

            csv_bytes = generate_csv_bytes(analysis)
            json_bytes = generate_json_bytes(analysis)
            txt_report = generate_text_report(analysis, transcript)

            st.download_button(
                "📥 Download CSV (Segment Data)",
                data=csv_bytes,
                file_name="callcenter_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.download_button(
                "📥 Download JSON (Summary)",
                data=json_bytes,
                file_name="callcenter_summary.json",
                mime="application/json",
                use_container_width=True,
            )

            st.download_button(
                "📥 Download Text Report",
                data=txt_report.encode("utf-8"),
                file_name="callcenter_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Segment Data Preview</div>', unsafe_allow_html=True)
            df_preview = pd.read_csv(
                __import__("io").StringIO(csv_bytes.decode("utf-8"))
            )
            st.dataframe(
                df_preview[["Speaker", "Text", "Sentiment", "Emotion"]].head(10),
                use_container_width=True,
                hide_index=True,
            )


    # ──────────────────────────────────────────────
    # History Panel
    # ──────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.divider()
        st.markdown('<div class="section-title">📚 Analysis History</div>', unsafe_allow_html=True)

        df_hist = pd.DataFrame(st.session_state.history)
        df_hist["agent_score"] = df_hist["agent_score"].apply(lambda x: f"{x:.0%}")
        df_hist["resolved"] = df_hist["resolved"].apply(lambda x: "✅" if x else "❌")
        df_hist.columns = ["Time", "Source", "Sentiment", "Agent Score", "Resolved"]
        st.dataframe(df_hist, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Landing state (no analysis yet)
# ─────────────────────────────────────────────────────────────────────────────

else:
    st.markdown("""
    <div class='info-banner' style='text-align:center; padding:40px 20px;'>
      <div style='font-size:3rem; margin-bottom:12px;'>🎙️</div>
      <div style='font-size:1.1rem; color:#334155; font-weight:600;'>
        Upload an interaction file or select a sample, then click <strong style='color:#1d4ed8;'>Analyze Interaction</strong>
      </div>
      <div style='font-size:0.85rem; color:#334155; margin-top:8px;'>
        Supports: Audio (.wav, .mp3, .m4a) · Transcript (.txt, .csv)
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature overview
    feat_cols = st.columns(4)
    features = [
        ("🧠", "3 NLP Models", "RoBERTa, BERT & DistilBERT ensemble for robust sentiment analysis"),
        ("😊", "Emotion Detection", "Identifies frustration, satisfaction, confusion, anger & more"),
        ("🎧", "Agent Assessment", "Scores empathy, professionalism & problem-solving behavior"),
        ("🎙️", "Speech-to-Text", "Auto-transcribes audio via OpenAI Whisper before analysis"),
    ]
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"""
            <div class='ai-card' style='text-align:center; padding:24px 16px;'>
              <div style='font-size:2rem; margin-bottom:10px;'>{icon}</div>
              <div style='font-weight:700; font-size:0.95rem; color:#0f172a; margin-bottom:6px;'>{title}</div>
              <div style='font-size:0.78rem; color:#334155; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
