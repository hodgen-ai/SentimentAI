# Vibemaxxing

**Optimize your vibes. Understand your environment. Level up your emotional awareness.**

Vibemaxxing is an always-on ambient sentiment analysis system that reveals the emotional dynamics of your daily conversations. It identifies speakers, tracks positivity patterns, and exposes insights like *"you're more negative after talking to Person X"*.

## Core Concept

The app runs quietly in the background:
- **Real-time vibe scoring** - Track emotional tone as conversations happen
- **Speaker identification** - Distinguish voices and track individual patterns
- **Relationship insights** - See how different people affect your mood
- **Pattern detection** - Find what times, places, and people correlate with your best vibes

## Key Features

### Ambient Listening
- Voice Activity Detection (VAD) triggers processing only when speech detected
- Battery-efficient for always-on use
- Local-only processing - nothing leaves your device

### Speaker Diarization
- Identifies and tracks multiple speakers
- Persistent speaker profiles across sessions
- Anonymous IDs (no names stored)

### Sentiment Analysis
- Positivity/negativity scoring (-1.0 to +1.0)
- Emotion classification (joy, sadness, anger, fear, surprise, neutral)
- Transformer-based accuracy (RoBERTa)

### Insight Generation
- "You tend to be more negative after talking to Speaker_3"
- "Your positivity peaks around 10:00 AM"
- "Your emotional baseline is trending upward this week"

## Technology Stack

| Component | Technology |
|-----------|------------|
| Speech Recognition | faster-whisper (4x faster than original) |
| Speaker Diarization | Pyannote.audio |
| Sentiment Analysis | RoBERTa (cardiffnlp) |
| Audio Capture | SoundDevice + WebRTC VAD |
| Storage | Local SQLite |

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo simulation
python demo.py

# Start real-time monitoring
python -m src.main start

# View statistics
python -m src.main stats

# Generate insights
python -m src.main insights
```

## Requirements

- Python 3.10+
- Microphone access
- ~2GB disk space for ML models
- HuggingFace token (for pyannote speaker diarization)

## Privacy First

- **100% local** - All processing on your device
- **No cloud** - No data uploaded anywhere
- **You control retention** - Auto-cleanup after configurable days
- **Anonymous speakers** - No names, just Speaker_1, Speaker_2, etc.

## Roadmap

- [x] Core sentiment analysis engine
- [ ] Real-time audio capture with VAD
- [ ] Speaker diarization integration
- [ ] CLI interface with live display
- [ ] Pattern detection algorithms
- [ ] Web dashboard (future)

## License

MIT License

## Disclaimer

This tool is for personal insight and self-improvement. Always obtain consent before recording others. Be mindful of privacy laws in your jurisdiction.
