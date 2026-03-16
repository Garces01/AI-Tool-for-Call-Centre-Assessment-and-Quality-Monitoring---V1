"""
Speech-to-Text Conversion Module
Handles audio file transcription using OpenAI Whisper
Supports: .wav, .mp3, .m4a, .ogg, .flac
"""

import os
import re
import json
import tempfile
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


class SpeechToTextConverter:
    """
    Converts audio files to text transcripts using OpenAI Whisper.
    Identifies speaker turns (agent vs customer) when possible.
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize the Whisper model.

        Args:
            model_size: One of 'tiny', 'base', 'small', 'medium', 'large'
                        'base' is recommended for balancing speed and accuracy.
        """
        self.model_size = model_size
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model lazily."""
        try:
            import whisper
            self.model = whisper.load_model(self.model_size)
            print(f"[STT] Whisper '{self.model_size}' model loaded successfully.")
        except ImportError:
            print("[STT] Whisper not installed. Falling back to mock transcription.")
            self.model = None
        except Exception as e:
            print(f"[STT] Failed to load Whisper model: {e}")
            self.model = None

    def convert_audio_to_text(self, audio_path: str) -> dict:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (.wav, .mp3, etc.)

        Returns:
            dict with keys: 'transcript', 'segments', 'language', 'duration'
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        supported = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4"}
        if audio_path.suffix.lower() not in supported:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}")

        # If Whisper is available, use it
        if self.model is not None:
            return self._transcribe_with_whisper(str(audio_path))

        # Otherwise return mock
        return self._mock_transcription(str(audio_path))

    def _transcribe_with_whisper(self, audio_path: str) -> dict:
        """Run Whisper transcription with word-level timestamps."""
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
                "speaker": self._guess_speaker(seg.get("start", 0.0)),
            })

        # Estimate duration from last segment end
        duration = segments[-1]["end"] if segments else 0.0

        return {
            "transcript": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language", "en"),
            "duration": duration,
            "source": "whisper",
        }

    def _guess_speaker(self, timestamp: float) -> str:
        """
        Simple heuristic: alternate speakers every ~30 seconds.
        In production, integrate a diarization model (e.g., pyannote.audio).
        """
        turn = int(timestamp // 30) % 2
        return "Agent" if turn == 0 else "Customer"

    def _mock_transcription(self, audio_path: str) -> dict:
        """
        Returns a realistic mock transcript when Whisper is unavailable.
        Useful for demos and testing.
        """
        mock_conversation = [
            {"start": 0.0, "end": 5.0, "text": "Thank you for calling TechSupport, my name is Sarah. How can I help you today?", "speaker": "Agent"},
            {"start": 5.5, "end": 12.0, "text": "Hi, I've been waiting for my order for two weeks now and I'm really frustrated. It was supposed to arrive last Monday.", "speaker": "Customer"},
            {"start": 12.5, "end": 18.0, "text": "I completely understand your frustration, and I sincerely apologize for the delay. Could I please have your order number so I can look into this right away?", "speaker": "Agent"},
            {"start": 18.5, "end": 23.0, "text": "It's 4521-XY. I need this urgently, it's for a birthday gift.", "speaker": "Customer"},
            {"start": 23.5, "end": 32.0, "text": "Thank you. I can see your order here. It looks like there was a delay at our fulfillment center. I'm going to escalate this and arrange express delivery at no extra charge. You should receive it within 48 hours.", "speaker": "Agent"},
            {"start": 32.5, "end": 38.0, "text": "Oh, that's great. Thank you for sorting that out. I feel much better now.", "speaker": "Customer"},
            {"start": 38.5, "end": 44.0, "text": "I'm so glad I could help. You'll receive a confirmation email with the new tracking number. Is there anything else I can assist you with today?", "speaker": "Agent"},
            {"start": 44.5, "end": 47.0, "text": "No, that's all. Thank you so much Sarah, really appreciate it.", "speaker": "Customer"},
            {"start": 47.5, "end": 52.0, "text": "My pleasure! Thank you for your patience and for calling TechSupport. Have a wonderful day!", "speaker": "Agent"},
        ]

        full_transcript = " ".join([s["text"] for s in mock_conversation])

        return {
            "transcript": full_transcript,
            "segments": mock_conversation,
            "language": "en",
            "duration": 52.0,
            "source": "mock",
        }

    def format_transcript_for_analysis(self, transcription_result: dict) -> str:
        """
        Format segments into a labeled transcript string.

        Returns:
            Multi-line string like:
            Agent: Hello, how can I help...
            Customer: I have a problem...
        """
        lines = []
        for seg in transcription_result.get("segments", []):
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def parse_transcript_file(self, file_path: str) -> dict:
        """
        Parse a text/CSV transcript file into the same structure as audio transcription.

        Supports formats:
          - Plain text with "Agent:" / "Customer:" prefixes
          - CSV with columns: speaker, text (optional: timestamp)
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".csv":
            return self._parse_csv_transcript(file_path)
        else:
            return self._parse_txt_transcript(file_path)

    def _parse_txt_transcript(self, file_path: Path) -> dict:
        """Parse a plain-text labeled transcript."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        segments = []
        lines = content.strip().split("\n")
        timestamp = 0.0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            speaker = "Unknown"
            text = line

            if line.lower().startswith("agent:"):
                speaker = "Agent"
                text = line[6:].strip()
            elif line.lower().startswith("customer:"):
                speaker = "Customer"
                text = line[9:].strip()

            duration = max(2.0, len(text.split()) * 0.4)
            segments.append({
                "start": timestamp,
                "end": timestamp + duration,
                "text": text,
                "speaker": speaker,
            })
            timestamp += duration + 0.5

        full_transcript = " ".join([s["text"] for s in segments])

        return {
            "transcript": full_transcript,
            "segments": segments,
            "language": "en",
            "duration": timestamp,
            "source": "text_file",
        }

    def _parse_csv_transcript(self, file_path: Path) -> dict:
        """Parse a CSV transcript with columns: speaker, text [, timestamp]."""
        import pandas as pd

        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]

        required = {"speaker", "text"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"CSV must have columns: {required}. Found: {list(df.columns)}")

        segments = []
        timestamp = 0.0

        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            speaker = str(row.get("speaker", "Unknown")).strip().title()
            duration = max(2.0, len(text.split()) * 0.4)

            segments.append({
                "start": float(row.get("timestamp", timestamp)),
                "end": timestamp + duration,
                "text": text,
                "speaker": speaker,
            })
            timestamp += duration + 0.5

        full_transcript = " ".join([s["text"] for s in segments])

        return {
            "transcript": full_transcript,
            "segments": segments,
            "language": "en",
            "duration": timestamp,
            "source": "csv_file",
        }
