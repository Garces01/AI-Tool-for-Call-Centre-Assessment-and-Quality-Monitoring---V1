"""
NLP Analysis Engine
Performs sentiment analysis and emotion detection using three transformer models:
  - RoBERTa  (cardiffnlp/twitter-roberta-base-sentiment-latest)
  - BERT     (nlptown/bert-base-multilingual-uncased-sentiment)
  - DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)

Also performs agent behavior classification (empathy, professionalism, problem-solving).
"""

import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class SentimentResult:
    label: str           # "positive" | "neutral" | "negative"
    score: float         # 0.0 – 1.0 confidence
    model: str

@dataclass
class EmotionResult:
    emotion: str         # e.g. "frustration", "satisfaction"
    score: float
    all_emotions: dict   # {emotion: score}

@dataclass
class AgentBehaviorResult:
    empathy_score: float          # 0–1
    professionalism_score: float  # 0–1
    problem_solving_score: float  # 0–1
    overall_score: float          # weighted average
    flags: list                   # list of positive/negative flags

@dataclass
class SegmentAnalysis:
    speaker: str
    text: str
    start: float
    end: float
    sentiment: SentimentResult
    emotion: Optional[EmotionResult] = None

@dataclass
class InteractionAnalysis:
    # Ensemble sentiment
    customer_overall_sentiment: str
    customer_sentiment_score: float

    # Per-segment analyses
    segment_analyses: list

    # Emotion
    customer_dominant_emotion: str
    customer_emotion_distribution: dict

    # Agent
    agent_behavior: AgentBehaviorResult

    # Resolution
    sentiment_after_resolution: str
    resolution_score: float
    issue_resolved: bool

    # Model breakdown
    roberta_result: SentimentResult
    bert_result: SentimentResult
    distilbert_result: SentimentResult

    # Metadata
    total_turns: int
    customer_turns: int
    agent_turns: int
    processing_time: float


# ─────────────────────────────────────────────
# Model Wrappers
# ─────────────────────────────────────────────

class RoBERTaAnalyzer:
    """
    Uses: cardiffnlp/twitter-roberta-base-sentiment-latest
    Labels: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
    """
    MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(self):
        self.pipeline = None
        self._load()

    def _load(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.MODEL_ID,
                truncation=True,
                max_length=512,
                top_k=None,
            )
            print("[RoBERTa] Model loaded.")
        except Exception as e:
            print(f"[RoBERTa] Load failed: {e} – using mock.")

    def analyze(self, text: str) -> SentimentResult:
        if not text.strip():
            return SentimentResult("neutral", 0.5, "roberta")

        if self.pipeline:
            try:
                results = self.pipeline(text[:512])[0]
                # results is list of {label, score}
                label_map = {
                    "LABEL_0": "negative",
                    "LABEL_1": "neutral",
                    "LABEL_2": "positive",
                }
                top = max(results, key=lambda x: x["score"])
                label = label_map.get(top["label"], top["label"].lower())
                return SentimentResult(label, top["score"], "roberta")
            except Exception as e:
                print(f"[RoBERTa] Inference error: {e}")

        return self._mock_analyze(text)

    def _mock_analyze(self, text: str) -> SentimentResult:
        score, label = _rule_based_sentiment(text)
        # Add slight model-specific noise
        noise = np.random.normal(0, 0.03)
        score = float(np.clip(score + noise, 0.0, 1.0))
        return SentimentResult(label, score, "roberta")


class BERTAnalyzer:
    """
    Uses: nlptown/bert-base-multilingual-uncased-sentiment
    Labels: 1-5 stars → mapped to negative/neutral/positive
    """
    MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"

    def __init__(self):
        self.pipeline = None
        self._load()

    def _load(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.MODEL_ID,
                truncation=True,
                max_length=512,
            )
            print("[BERT] Model loaded.")
        except Exception as e:
            print(f"[BERT] Load failed: {e} – using mock.")

    def analyze(self, text: str) -> SentimentResult:
        if not text.strip():
            return SentimentResult("neutral", 0.5, "bert")

        if self.pipeline:
            try:
                result = self.pipeline(text[:512])[0]
                stars = int(result["label"].split()[0])
                score = result["score"]
                if stars <= 2:
                    label = "negative"
                elif stars == 3:
                    label = "neutral"
                else:
                    label = "positive"
                return SentimentResult(label, score, "bert")
            except Exception as e:
                print(f"[BERT] Inference error: {e}")

        return self._mock_analyze(text)

    def _mock_analyze(self, text: str) -> SentimentResult:
        score, label = _rule_based_sentiment(text)
        noise = np.random.normal(0, 0.04)
        score = float(np.clip(score + noise, 0.0, 1.0))
        return SentimentResult(label, score, "bert")


class DistilBERTAnalyzer:
    """
    Uses: distilbert-base-uncased-finetuned-sst-2-english
    Labels: POSITIVE / NEGATIVE (binary)
    """
    MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        self.pipeline = None
        self._load()

    def _load(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.MODEL_ID,
                truncation=True,
                max_length=512,
                top_k=None,
            )
            print("[DistilBERT] Model loaded.")
        except Exception as e:
            print(f"[DistilBERT] Load failed: {e} – using mock.")

    def analyze(self, text: str) -> SentimentResult:
        if not text.strip():
            return SentimentResult("neutral", 0.5, "distilbert")

        if self.pipeline:
            try:
                results = self.pipeline(text[:512])[0]
                top = max(results, key=lambda x: x["score"])
                label = top["label"].lower()  # "positive" or "negative"
                # DistilBERT has no "neutral" – use confidence threshold
                if top["score"] < 0.70:
                    label = "neutral"
                return SentimentResult(label, top["score"], "distilbert")
            except Exception as e:
                print(f"[DistilBERT] Inference error: {e}")

        return self._mock_analyze(text)

    def _mock_analyze(self, text: str) -> SentimentResult:
        score, label = _rule_based_sentiment(text)
        noise = np.random.normal(0, 0.03)
        score = float(np.clip(score + noise, 0.0, 1.0))
        return SentimentResult(label, score, "distilbert")


# ─────────────────────────────────────────────
# Emotion Detector
# ─────────────────────────────────────────────

class EmotionDetector:
    """
    Detects customer emotions from text.
    Tries: j-hartmann/emotion-english-distilroberta-base (HuggingFace)
    Falls back to lexicon-based approach.
    """
    MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

    EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
                "frustration", "confusion", "satisfaction", "anxiety"]

    # Mapped emotions for display
    DISPLAY_MAP = {
        "anger": "Anger",
        "disgust": "Disgust",
        "fear": "Fear/Anxiety",
        "joy": "Joy/Satisfaction",
        "neutral": "Neutral",
        "sadness": "Sadness",
        "surprise": "Surprise",
        "frustration": "Frustration",
        "confusion": "Confusion",
        "satisfaction": "Satisfaction",
        "anxiety": "Anxiety",
    }

    def __init__(self):
        self.pipeline = None
        self._load()

    def _load(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.MODEL_ID,
                truncation=True,
                max_length=512,
                top_k=None,
            )
            print("[Emotion] Model loaded.")
        except Exception as e:
            print(f"[Emotion] Load failed: {e} – using lexicon approach.")

    def detect(self, text: str) -> EmotionResult:
        if not text.strip():
            return EmotionResult("neutral", 1.0, {"neutral": 1.0})

        if self.pipeline:
            try:
                results = self.pipeline(text[:512])[0]
                scores = {r["label"].lower(): r["score"] for r in results}
                top_emotion = max(scores, key=scores.get)
                return EmotionResult(
                    self.DISPLAY_MAP.get(top_emotion, top_emotion),
                    scores[top_emotion],
                    {self.DISPLAY_MAP.get(k, k): v for k, v in scores.items()},
                )
            except Exception as e:
                print(f"[Emotion] Inference error: {e}")

        return self._lexicon_detect(text)

    def _lexicon_detect(self, text: str) -> EmotionResult:
        """Rule-based emotion detection using keyword lexicons."""
        text_l = text.lower()

        lexicons = {
            "Frustration": [
                "frustrated", "annoying", "irritating", "ridiculous", "unacceptable",
                "terrible", "awful", "horrible", "outrageous", "useless", "worst",
                "not working", "keeps happening", "same issue", "still waiting",
            ],
            "Anger": [
                "angry", "furious", "rage", "demand", "lawsuit", "compensation",
                "disgusting", "incompetent", "never again", "cancel", "scam",
            ],
            "Confusion": [
                "confused", "don't understand", "not sure", "unclear", "what do you mean",
                "how does", "explain", "lost", "don't know", "help me understand",
            ],
            "Satisfaction": [
                "thank you", "thanks", "great", "wonderful", "perfect", "excellent",
                "appreciate", "helpful", "amazing", "happy", "pleased", "relieved",
                "problem solved", "much better", "glad",
            ],
            "Fear/Anxiety": [
                "worried", "concern", "scared", "afraid", "nervous", "anxious",
                "urgent", "emergency", "important", "need immediately",
            ],
            "Sadness": [
                "disappointed", "let down", "sad", "unhappy", "expected better",
                "hoped", "wish", "regret",
            ],
            "Neutral": ["okay", "fine", "alright", "sure", "yes", "no", "maybe"],
        }

        scores = {}
        for emotion, keywords in lexicons.items():
            count = sum(1 for kw in keywords if kw in text_l)
            scores[emotion] = min(count * 0.25, 1.0)

        if max(scores.values()) == 0:
            scores["Neutral"] = 1.0

        # Normalize
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        top = max(scores, key=scores.get)

        return EmotionResult(top, scores[top], scores)


# ─────────────────────────────────────────────
# Agent Behavior Analyzer
# ─────────────────────────────────────────────

class AgentBehaviorAnalyzer:
    """
    Analyzes agent messages for:
    - Empathy (acknowledging customer feelings)
    - Professionalism (tone, courtesy)
    - Problem-solving (actionable responses)
    """

    EMPATHY_PHRASES = [
        "i understand", "i'm sorry", "i apologize", "i sincerely apologize",
        "i can imagine", "that must be", "i hear you", "completely understand",
        "i appreciate your patience", "sorry to hear", "i see how", "understandably",
        "your frustration", "i feel for you", "must be frustrating",
    ]

    PROFESSIONALISM_PHRASES = [
        "certainly", "of course", "absolutely", "please", "thank you",
        "my pleasure", "happy to help", "right away", "i'll take care",
        "allow me to", "one moment please", "let me check", "i assure you",
        "rest assured", "we value", "for your convenience",
    ]

    PROBLEM_SOLVING_PHRASES = [
        "i will", "i'll", "let me", "i can", "we can", "i'm going to",
        "i've escalated", "arranged", "processed", "resolved", "fixed",
        "solution", "here's what", "i found", "looks like", "i've located",
        "tracking number", "refund", "replacement", "expedite", "within",
    ]

    NEGATIVE_FLAGS = [
        ("Used dismissive language", ["that's not my problem", "nothing i can do", "not our fault", "read the policy"]),
        ("Interrupted customer", ["but wait", "no no", "wrong wrong"]),
        ("Made promises without action", ["i promise", "i guarantee"]),
    ]

    def analyze(self, agent_segments: list) -> AgentBehaviorResult:
        if not agent_segments:
            return AgentBehaviorResult(0.0, 0.0, 0.0, 0.0, ["No agent turns found"])

        combined = " ".join([s["text"].lower() for s in agent_segments])

        empathy = self._score_phrases(combined, self.EMPATHY_PHRASES)
        professionalism = self._score_phrases(combined, self.PROFESSIONALISM_PHRASES)
        problem_solving = self._score_phrases(combined, self.PROBLEM_SOLVING_PHRASES)

        flags = []
        for flag_name, phrases in self.NEGATIVE_FLAGS:
            if any(p in combined for p in phrases):
                flags.append(f"⚠️ {flag_name}")

        if empathy > 0.6:
            flags.append("✅ Demonstrated empathy")
        if professionalism > 0.6:
            flags.append("✅ Maintained professionalism")
        if problem_solving > 0.6:
            flags.append("✅ Proactive problem-solving")
        if not flags:
            flags.append("ℹ️ Standard interaction – no notable behaviors flagged")

        overall = (empathy * 0.35 + professionalism * 0.35 + problem_solving * 0.30)

        return AgentBehaviorResult(
            empathy_score=empathy,
            professionalism_score=professionalism,
            problem_solving_score=problem_solving,
            overall_score=overall,
            flags=flags,
        )

    def _score_phrases(self, text: str, phrases: list) -> float:
        matches = sum(1 for p in phrases if p in text)
        # Normalize: 3+ matches = full score
        return min(matches / max(len(phrases) * 0.15, 1.0), 1.0)


# ─────────────────────────────────────────────
# Ensemble + Main Analyzer
# ─────────────────────────────────────────────

def _rule_based_sentiment(text: str) -> tuple:
    """Simple rule-based fallback sentiment scoring."""
    POSITIVE = [
        "thank", "great", "excellent", "happy", "satisfied", "wonderful", "perfect",
        "appreciate", "glad", "pleased", "amazing", "good", "nice", "helpful",
        "resolved", "fixed", "better", "love", "fantastic", "appreciate",
    ]
    NEGATIVE = [
        "frustrated", "angry", "terrible", "horrible", "awful", "upset", "bad",
        "disappointed", "unacceptable", "worst", "ridiculous", "annoying",
        "problem", "issue", "wrong", "broken", "doesn't work", "never",
    ]

    text_l = text.lower()
    pos = sum(1 for w in POSITIVE if w in text_l)
    neg = sum(1 for w in NEGATIVE if w in text_l)

    total = pos + neg
    if total == 0:
        return 0.5, "neutral"

    ratio = pos / total
    if ratio >= 0.65:
        return ratio, "positive"
    elif ratio <= 0.35:
        return 1 - ratio, "negative"
    else:
        return 0.5, "neutral"


def ensemble_sentiment(results: list) -> tuple:
    """
    Combine multiple SentimentResult objects via weighted voting.
    Returns (label, confidence_score)
    """
    weights = {"roberta": 0.40, "bert": 0.35, "distilbert": 0.25}
    scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    for r in results:
        w = weights.get(r.model, 0.33)
        label = r.label if r.label in scores else "neutral"
        scores[label] += w * r.score

    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    label = max(scores, key=scores.get)
    return label, scores[label]


class CallCenterAnalyzer:
    """
    Main orchestrator: loads models and runs full interaction analysis.
    """

    def __init__(self):
        print("[Analyzer] Initializing models...")
        self.roberta = RoBERTaAnalyzer()
        self.bert = BERTAnalyzer()
        self.distilbert = DistilBERTAnalyzer()
        self.emotion_detector = EmotionDetector()
        self.agent_analyzer = AgentBehaviorAnalyzer()
        print("[Analyzer] Ready.")

    def analyze_interaction(self, transcription_result: dict) -> InteractionAnalysis:
        """
        Full analysis pipeline.

        Args:
            transcription_result: dict from SpeechToTextConverter with 'segments' key

        Returns:
            InteractionAnalysis dataclass
        """
        start_time = time.time()

        segments = transcription_result.get("segments", [])
        if not segments:
            raise ValueError("No segments found in transcription result.")

        customer_segments = [s for s in segments if s.get("speaker", "").lower() == "customer"]
        agent_segments = [s for s in segments if s.get("speaker", "").lower() == "agent"]

        # ── Per-segment analysis ──
        segment_analyses = []
        for seg in segments:
            text = seg.get("text", "")
            rob = self.roberta.analyze(text)
            ber = self.bert.analyze(text)
            dis = self.distilbert.analyze(text)
            ens_label, ens_score = ensemble_sentiment([rob, ber, dis])

            emo = None
            if seg.get("speaker", "").lower() == "customer":
                emo = self.emotion_detector.detect(text)

            segment_analyses.append(SegmentAnalysis(
                speaker=seg.get("speaker", "Unknown"),
                text=text,
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                sentiment=SentimentResult(ens_label, ens_score, "ensemble"),
                emotion=emo,
            ))

        # ── Customer overall sentiment (ensemble across all customer turns) ──
        customer_text = " ".join([s["text"] for s in customer_segments])
        rob_overall = self.roberta.analyze(customer_text)
        bert_overall = self.bert.analyze(customer_text)
        dis_overall = self.distilbert.analyze(customer_text)
        overall_label, overall_score = ensemble_sentiment([rob_overall, bert_overall, dis_overall])

        # ── Customer emotion distribution ──
        all_emotions = {}
        for seg in customer_segments:
            emo = self.emotion_detector.detect(seg["text"])
            for emotion, score in emo.all_emotions.items():
                all_emotions[emotion] = all_emotions.get(emotion, 0.0) + score

        total_emo = sum(all_emotions.values()) or 1.0
        emotion_distribution = {k: v / total_emo for k, v in sorted(all_emotions.items(), key=lambda x: -x[1])}
        dominant_emotion = max(emotion_distribution, key=emotion_distribution.get) if emotion_distribution else "Neutral"

        # ── Agent behavior ──
        agent_behavior = self.agent_analyzer.analyze(agent_segments)

        # ── Sentiment after resolution (last 2 customer segments) ──
        last_customer_segs = customer_segments[-2:] if len(customer_segments) >= 2 else customer_segments
        resolution_text = " ".join([s["text"] for s in last_customer_segs])
        r1 = self.roberta.analyze(resolution_text)
        r2 = self.bert.analyze(resolution_text)
        r3 = self.distilbert.analyze(resolution_text)
        res_label, res_score = ensemble_sentiment([r1, r2, r3])
        issue_resolved = res_label == "positive" and res_score > 0.55

        return InteractionAnalysis(
            customer_overall_sentiment=overall_label,
            customer_sentiment_score=overall_score,
            segment_analyses=segment_analyses,
            customer_dominant_emotion=dominant_emotion,
            customer_emotion_distribution=emotion_distribution,
            agent_behavior=agent_behavior,
            sentiment_after_resolution=res_label,
            resolution_score=res_score,
            issue_resolved=issue_resolved,
            roberta_result=rob_overall,
            bert_result=bert_overall,
            distilbert_result=dis_overall,
            total_turns=len(segments),
            customer_turns=len(customer_segments),
            agent_turns=len(agent_segments),
            processing_time=time.time() - start_time,
        )
