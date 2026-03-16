"""
Report Generator
Exports interaction analysis results to PDF and CSV formats.
"""

import io
import json
import datetime
import pandas as pd
from pathlib import Path


def analysis_to_dataframe(analysis) -> pd.DataFrame:
    """
    Convert segment-level analysis to a Pandas DataFrame for CSV export.
    """
    rows = []
    for seg in analysis.segment_analyses:
        row = {
            "Timestamp_Start": round(seg.start, 2),
            "Timestamp_End": round(seg.end, 2),
            "Speaker": seg.speaker,
            "Text": seg.text,
            "Sentiment": seg.sentiment.label.capitalize(),
            "Sentiment_Confidence": round(seg.sentiment.score, 3),
            "Emotion": seg.emotion.emotion if seg.emotion else "N/A",
            "Emotion_Confidence": round(seg.emotion.score, 3) if seg.emotion else "N/A",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def generate_summary_dict(analysis) -> dict:
    """
    Convert InteractionAnalysis to a JSON-serializable summary dictionary.
    """
    return {
        "analysis_timestamp": datetime.datetime.now().isoformat(),
        "customer_overall_sentiment": analysis.customer_overall_sentiment,
        "customer_sentiment_score": round(analysis.customer_sentiment_score, 3),
        "customer_dominant_emotion": analysis.customer_dominant_emotion,
        "emotion_distribution": {k: round(v, 3) for k, v in analysis.customer_emotion_distribution.items()},
        "agent_performance": {
            "overall_score": round(analysis.agent_behavior.overall_score, 3),
            "empathy": round(analysis.agent_behavior.empathy_score, 3),
            "professionalism": round(analysis.agent_behavior.professionalism_score, 3),
            "problem_solving": round(analysis.agent_behavior.problem_solving_score, 3),
            "flags": analysis.agent_behavior.flags,
        },
        "resolution": {
            "sentiment_after_resolution": analysis.sentiment_after_resolution,
            "resolution_score": round(analysis.resolution_score, 3),
            "issue_resolved": analysis.issue_resolved,
        },
        "model_breakdown": {
            "roberta": {"label": analysis.roberta_result.label, "score": round(analysis.roberta_result.score, 3)},
            "bert": {"label": analysis.bert_result.label, "score": round(analysis.bert_result.score, 3)},
            "distilbert": {"label": analysis.distilbert_result.label, "score": round(analysis.distilbert_result.score, 3)},
        },
        "interaction_stats": {
            "total_turns": analysis.total_turns,
            "customer_turns": analysis.customer_turns,
            "agent_turns": analysis.agent_turns,
            "processing_time_seconds": round(analysis.processing_time, 2),
        },
    }


def generate_csv_bytes(analysis) -> bytes:
    """Return CSV bytes of segment-level analysis."""
    df = analysis_to_dataframe(analysis)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def generate_json_bytes(analysis) -> bytes:
    """Return JSON bytes of summary."""
    summary = generate_summary_dict(analysis)
    return json.dumps(summary, indent=2).encode("utf-8")


def generate_text_report(analysis, transcript_result: dict) -> str:
    """
    Generate a plain-text report for display or download.
    """
    lines = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 65)
    lines.append("    CALL CENTER INTERACTION ANALYSIS REPORT")
    lines.append(f"    Generated: {now}")
    lines.append("=" * 65)
    lines.append("")

    # ── Summary ──
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(
        analysis.customer_overall_sentiment, "❓"
    )
    lines.append(f"Customer Overall Sentiment : {analysis.customer_overall_sentiment.upper()} {sentiment_emoji} ({analysis.customer_sentiment_score:.0%})")
    lines.append(f"Dominant Customer Emotion  : {analysis.customer_dominant_emotion}")
    lines.append(f"Issue Resolved             : {'YES ✅' if analysis.issue_resolved else 'NO ❌'}")
    lines.append(f"Post-Resolution Sentiment  : {analysis.sentiment_after_resolution.upper()}")
    lines.append(f"Agent Performance Score    : {analysis.agent_behavior.overall_score:.0%}")
    lines.append("")

    # ── Model Comparison ──
    lines.append("MODEL BREAKDOWN (Sentiment)")
    lines.append("-" * 40)
    lines.append(f"  RoBERTa    : {analysis.roberta_result.label.capitalize()} ({analysis.roberta_result.score:.0%})")
    lines.append(f"  BERT       : {analysis.bert_result.label.capitalize()} ({analysis.bert_result.score:.0%})")
    lines.append(f"  DistilBERT : {analysis.distilbert_result.label.capitalize()} ({analysis.distilbert_result.score:.0%})")
    lines.append("")

    # ── Agent Behavior ──
    lines.append("AGENT PERFORMANCE")
    lines.append("-" * 40)
    lines.append(f"  Empathy          : {analysis.agent_behavior.empathy_score:.0%}")
    lines.append(f"  Professionalism  : {analysis.agent_behavior.professionalism_score:.0%}")
    lines.append(f"  Problem-Solving  : {analysis.agent_behavior.problem_solving_score:.0%}")
    lines.append(f"  Overall Score    : {analysis.agent_behavior.overall_score:.0%}")
    lines.append("")
    lines.append("  Behavior Flags:")
    for flag in analysis.agent_behavior.flags:
        lines.append(f"    {flag}")
    lines.append("")

    # ── Emotion Distribution ──
    lines.append("CUSTOMER EMOTION DISTRIBUTION")
    lines.append("-" * 40)
    for emotion, score in list(analysis.customer_emotion_distribution.items())[:5]:
        bar = "█" * int(score * 20)
        lines.append(f"  {emotion:<18}: {bar} {score:.0%}")
    lines.append("")

    # ── Conversation Transcript ──
    lines.append("CONVERSATION TRANSCRIPT")
    lines.append("-" * 40)
    for seg in transcript_result.get("segments", []):
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        lines.append(f"  [{speaker}]: {text}")
    lines.append("")
    lines.append("=" * 65)
    lines.append("END OF REPORT")
    lines.append("=" * 65)

    return "\n".join(lines)
