# 🎙️ Call Center AI Quality Monitor

An intelligent, automated tool for assessing and monitoring call center agent performance using Natural Language Processing. Combines three transformer models (RoBERTa, BERT, DistilBERT) for robust sentiment analysis, emotion detection, and agent behavior scoring.

---

## 📋 Features

| Feature | Details |
|---|---|
| **Sentiment Analysis** | Ensemble of RoBERTa + BERT + DistilBERT models |
| **Emotion Detection** | Frustration, Anger, Satisfaction, Confusion, Anxiety & more |
| **Agent Assessment** | Empathy, Professionalism, Problem-Solving scores |
| **Speech-to-Text** | OpenAI Whisper auto-transcription for audio files |
| **Interactive Dashboard** | Streamlit app with real-time Plotly visualizations |
| **Export** | CSV, JSON, and plain-text report downloads |

---

## 🚀 Quick Start

### 1. Clone / Download the project

```bash
cd callcenter_ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **PyTorch note**: For CPU-only machines, install PyTorch first:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```
> For GPU (CUDA 12.1):
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📁 Project Structure

```
callcenter_ai/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/
│   ├── __init__.py
│   └── analyzer.py                 # Core NLP analysis engine
│       ├── RoBERTaAnalyzer         # cardiffnlp/twitter-roberta-base-sentiment-latest
│       ├── BERTAnalyzer            # nlptown/bert-base-multilingual-uncased-sentiment
│       ├── DistilBERTAnalyzer      # distilbert-base-uncased-finetuned-sst-2-english
│       ├── EmotionDetector         # j-hartmann/emotion-english-distilroberta-base
│       ├── AgentBehaviorAnalyzer   # Rule-based + NLP agent scoring
│       └── CallCenterAnalyzer      # Main orchestrator
│
├── utils/
│   ├── __init__.py
│   ├── speech_to_text.py           # Whisper-based audio transcription + file parsers
│   └── report_generator.py         # CSV / JSON / TXT export utilities
│
└── sample_data/
    ├── sample_transcript_1.txt     # Tech support – internet outage
    ├── sample_transcript_2.txt     # Banking – billing dispute
    └── sample_transcript_3.csv     # Pharmacy – prescription inquiry
```

---

## 🧠 Models Used

### Sentiment Analysis (Ensemble)

| Model | HuggingFace ID | Weight |
|---|---|---|
| **RoBERTa** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 40% |
| **BERT** | `nlptown/bert-base-multilingual-uncased-sentiment` | 35% |
| **DistilBERT** | `distilbert-base-uncased-finetuned-sst-2-english` | 25% |

The three models vote with weighted confidence scores. Final label = highest weighted sum.

### Emotion Detection

| Model | HuggingFace ID |
|---|---|
| **DistilRoBERTa** | `j-hartmann/emotion-english-distilroberta-base` |

Detects: anger, disgust, fear, joy, neutral, sadness, surprise — mapped to call center emotions.

### Speech-to-Text

| Component | Details |
|---|---|
| **Whisper** | `openai/whisper-base` (configurable: tiny/base/small/medium/large) |

---

## 📥 Input Formats

### Audio Files
Supported: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`
- Auto-transcribed using Whisper
- Speaker diarization: heuristic alternation (upgrade with pyannote.audio for production)

### Text Transcripts — TXT
```
Agent: Thank you for calling, how can I help you?
Customer: I have a problem with my order.
Agent: I'd be happy to help. Can I have your order number?
```

### Text Transcripts — CSV
```csv
speaker,text,timestamp
Agent,Thank you for calling,0
Customer,I have a problem,5
Agent,I'd be happy to help,10
```

---

## 📊 Dashboard Tabs

| Tab | Contents |
|---|---|
| **Sentiment Trends** | Timeline chart, model comparison, distribution bar |
| **Emotion Analysis** | Pie chart, bubble scatter over time, intensity heatmap |
| **Agent Performance** | Radar chart, behavior scores, resolution gauge |
| **Transcript** | Color-coded annotated conversation with per-turn metrics |
| **Summary & Export** | Full metrics table, CSV/JSON/TXT downloads |

---

## ⚙️ Configuration

Edit the sidebar at runtime, or modify defaults in `app.py`:

```python
whisper_model = "base"   # tiny | base | small | medium | large
run_emotion = True       # Enable/disable emotion detection
run_agent = True         # Enable/disable agent behavior analysis
```

---

## 🔧 Extending the System

### Add a new model
```python
# In models/analyzer.py
class MyCustomAnalyzer:
    def analyze(self, text: str) -> SentimentResult:
        # your inference code
        return SentimentResult(label, score, "my_model")
```

Then add it to the `ensemble_sentiment()` weights dict.

### Add speaker diarization
Replace `_guess_speaker()` in `utils/speech_to_text.py` with:
```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
```

### Batch processing
```python
from models.analyzer import CallCenterAnalyzer
from utils.speech_to_text import SpeechToTextConverter

analyzer = CallCenterAnalyzer()
stt = SpeechToTextConverter()

for file in Path("recordings/").glob("*.wav"):
    transcript = stt.convert_audio_to_text(str(file))
    result = analyzer.analyze_interaction(transcript)
    print(f"{file.name}: {result.customer_overall_sentiment} | Agent: {result.agent_behavior.overall_score:.0%}")
```

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.32 | Dashboard framework |
| `transformers` | ≥4.38 | HuggingFace transformer models |
| `torch` | ≥2.1 | PyTorch backend |
| `openai-whisper` | latest | Speech-to-text |
| `plotly` | ≥5.18 | Interactive charts |
| `pandas` | ≥2.1 | Data processing |

---

## 🔒 Privacy Note

All processing is performed **locally** on your machine. No audio or transcript data is sent to external APIs. HuggingFace model weights are downloaded once and cached in `~/.cache/huggingface/`.

---

## 📄 License

MIT License — free for academic and commercial use.
