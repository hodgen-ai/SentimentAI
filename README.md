# SentimentAI

## Product Vision

SentimentAI is an always-on ambient sentiment analysis system that helps you understand the emotional dynamics of your environment. By continuously listening to conversations around you, it provides insights into the positivity and negativity patterns in your daily interactions.

## Core Concept

The app runs quietly in the background, analyzing:
- **Real-time sentiment scoring** - Track the emotional tone of conversations as they happen
- **Speaker identification** - Distinguish between different voices and track individual patterns
- **Relationship insights** - Discover how different people and contexts affect your mood
- **Temporal patterns** - Identify times of day, days of week, or situations that correlate with emotional states

## Key Features

### Ambient Listening
- Continuous audio monitoring with privacy-first design
- Local processing - no data leaves your device
- Automatic conversation detection and segmentation

### Speaker Diarization
- Identifies and tracks multiple speakers in a conversation
- Builds anonymous speaker profiles over time
- No facial recognition or personally identifiable data storage

### Sentiment Analysis
- Real-time positivity/negativity scoring (-1.0 to +1.0 scale)
- Emotion classification (joy, sadness, anger, fear, surprise, neutral)
- Confidence metrics for each analysis

### Insight Generation
- "You tend to be more negative after talking to Person X"
- "Your positivity peaks on Tuesday mornings"
- "Conversations in Location Y are 30% more positive than average"
- "Your emotional baseline is trending upward this month"

## Use Cases

1. **Personal Development** - Identify toxic relationships or situations that drain your energy
2. **Workplace Dynamics** - Understand team morale and meeting effectiveness
3. **Mental Health Tracking** - Monitor emotional trends and correlate with life events
4. **Social Research** - Analyze conversation patterns in different settings

## Technology Stack

- **Speech Recognition**: OpenAI Whisper for transcription
- **Speaker Diarization**: Pyannote.audio for speaker identification
- **Sentiment Analysis**: Transformer-based models (DistilBERT, RoBERTa)
- **Audio Processing**: Librosa, SoundDevice for real-time capture
- **Data Storage**: Local SQLite database with encryption

## Privacy & Ethics

- All processing happens locally on your device
- No cloud uploads or third-party data sharing
- User-controlled retention policies
- Anonymous speaker IDs (no names or identifying info)
- Easy pause/delete functionality
- Transparent about what data is collected and how it's used

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo simulation
python demo.py

# (Future) Start real-time monitoring
python src/main.py --mode realtime
```

## Roadmap

- [ ] Phase 1: Core sentiment analysis engine
- [ ] Phase 2: Real-time audio capture and processing
- [ ] Phase 3: Speaker diarization integration
- [ ] Phase 4: Web dashboard for insights visualization
- [ ] Phase 5: Mobile app companion
- [ ] Phase 6: Advanced pattern recognition and predictions

## License

MIT License - See LICENSE file for details

## Disclaimer

This tool is designed for personal insight and self-improvement. Always obtain consent before recording conversations with others. Be mindful of privacy laws and regulations in your jurisdiction.
