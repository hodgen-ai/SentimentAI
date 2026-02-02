#!/usr/bin/env python3
"""
Vibemaxxing Demo - Simulates a day of conversations with sentiment analysis.

This demo shows how Vibemaxxing tracks conversations throughout a day,
identifying speakers and analyzing vibe patterns.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.sentiment import SentimentAnalyzer, ConversationAnalytics, Emotion


# Simulated conversation data for a typical day
CONVERSATIONS = [
    # Morning - Coffee with roommate (generally positive)
    ("Roommate", "Good morning! Did you sleep well?"),
    ("You", "Yeah, pretty good. Coffee smells amazing."),
    ("Roommate", "I tried that new blend. It's fantastic!"),
    ("You", "Love it. This is exactly what I needed."),

    # Morning - Stand-up meeting with team (neutral/professional)
    ("Manager", "Alright team, let's start with updates. Who wants to go first?"),
    ("You", "I can start. Finished the API integration yesterday."),
    ("Colleague_A", "Great work! That unblocks me for the frontend."),
    ("Manager", "Excellent. Any blockers we should know about?"),
    ("You", "Nope, all good. Should be smooth from here."),

    # Late Morning - 1:1 with difficult client (tense)
    ("Client_X", "I'm really frustrated with these delays. This is unacceptable."),
    ("You", "I understand your concerns. Let me explain what happened."),
    ("Client_X", "I don't want excuses. I want results. This is disappointing."),
    ("You", "I hear you. We're working hard to resolve this. I'm sorry for the delay."),
    ("Client_X", "This better be fixed by tomorrow. I'm not happy about this."),
    ("You", "We'll get it done. I'll send you an update by end of day."),

    # Lunch - Casual chat with friend (positive, relaxed)
    ("Friend_B", "Hey! So good to see you. How's everything going?"),
    ("You", "Much better now that I'm out of that meeting. Rough morning."),
    ("Friend_B", "Ugh, those are the worst. Let's grab something delicious."),
    ("You", "Perfect. I'm thinking tacos. That place on 5th is amazing."),
    ("Friend_B", "Yes! Their salsa is incredible. This is going to be great."),

    # Afternoon - Brainstorming session (creative, energetic)
    ("Colleague_A", "What if we completely redesign the user flow?"),
    ("You", "I love that idea! We could make it so much more intuitive."),
    ("Colleague_C", "That's brilliant. And we could add animations here..."),
    ("You", "Yes! This is exciting. Let's prototype it this week."),
    ("Colleague_A", "I'm so pumped about this. Best idea we've had in months."),

    # Late Afternoon - Check-in with micromanaging boss (stressful)
    ("Boss_Y", "I need to see your progress report. Why isn't it done yet?"),
    ("You", "I'm working on it now. Should have it to you in an hour."),
    ("Boss_Y", "An hour? I needed this this morning. This is frustrating."),
    ("You", "I apologize. I had the client call this morning. It took longer than expected."),
    ("Boss_Y", "I don't care about excuses. Just get it done. I'm disappointed."),
    ("You", "Understood. I'll prioritize it right now."),

    # Evening - Call with supportive partner (warm, positive)
    ("Partner", "Hey love! How was your day? You sound tired."),
    ("You", "It was rough. Had some difficult conversations today."),
    ("Partner", "I'm sorry, that sounds stressful. Want to talk about it?"),
    ("You", "Thanks. I appreciate you. Just glad to hear your voice."),
    ("Partner", "You're amazing. You handled it. I'm proud of you."),
    ("You", "You always know what to say. I love you."),
    ("Partner", "Love you too. Let's do something fun this weekend, okay?"),
    ("You", "Perfect. That sounds wonderful. Can't wait."),

    # Night - Gaming with online friends (fun, casual)
    ("Gamer_D", "Dude, that was an insane play! You're crushing it tonight!"),
    ("You", "Haha thanks! I'm finally in the zone. This is so much fun."),
    ("Gamer_E", "We're unstoppable! Best team ever!"),
    ("You", "This is exactly what I needed. So much better than this morning."),
    ("Gamer_D", "Game nights are the best. Same time tomorrow?"),
    ("You", "Absolutely. This is awesome. See you tomorrow!"),
]


def simulate_day():
    """Simulate a full day of conversations with sentiment analysis."""

    print("=" * 80)
    print("Vibemaxxing - Day Simulation Demo")
    print("=" * 80)
    print()
    print("Analyzing a day of conversations with sentiment tracking...")
    print()

    # Initialize analyzer and analytics
    analyzer = SentimentAnalyzer()
    analytics = ConversationAnalytics()

    # Starting time (8 AM)
    current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)

    # Process conversations
    print("Conversation Timeline:")
    print("-" * 80)

    last_context = ""
    for speaker, text in CONVERSATIONS:
        # Determine context based on speaker pattern
        if "Roommate" in speaker:
            context = "☕ Morning Coffee"
        elif "Manager" in speaker or "Colleague" in speaker:
            context = "💼 Work Meeting"
        elif "Client" in speaker:
            context = "📞 Client Call"
        elif "Friend" in speaker:
            context = "🍽️  Lunch Break"
        elif "Boss" in speaker:
            context = "😰 Boss Check-in"
        elif "Partner" in speaker:
            context = "💕 Evening Call"
        elif "Gamer" in speaker:
            context = "🎮 Gaming Session"
        else:
            context = "💬 Conversation"

        # Print context header if changed
        if context != last_context:
            if last_context:
                print()
            print(f"\n[{current_time.strftime('%I:%M %p')}] {context}")
            last_context = context

        # Analyze sentiment
        score = analyzer.analyze(text, speaker_id=speaker)
        analytics.add_utterance(score)

        # Display with color coding
        sentiment_indicator = "😊" if score.sentiment > 0.3 else "😔" if score.sentiment < -0.3 else "😐"
        emotion_emoji = {
            Emotion.JOY: "😄",
            Emotion.SADNESS: "😢",
            Emotion.ANGER: "😠",
            Emotion.FEAR: "😰",
            Emotion.SURPRISE: "😲",
            Emotion.NEUTRAL: "😐"
        }

        print(f"  {sentiment_indicator} {speaker:15s} ({score.sentiment:+.2f}): {text[:60]}...")

        # Advance time randomly (1-15 minutes per utterance)
        current_time += timedelta(minutes=random.randint(1, 15))

    print("\n" + "=" * 80)
    print("End of Day Analysis")
    print("=" * 80)
    print()

    # Speaker sentiment summary
    print("📊 Sentiment by Speaker:")
    print("-" * 80)

    speaker_rankings = analytics.compare_speakers()
    for rank, (speaker, avg_sentiment) in enumerate(speaker_rankings, 1):
        profile = analytics.get_speaker_profile(speaker)

        # Calculate sentiment bar
        bar_length = int((avg_sentiment + 1) * 20)  # Scale -1 to 1 -> 0 to 40
        bar = "█" * bar_length + "░" * (40 - bar_length)

        sentiment_label = "POSITIVE" if avg_sentiment > 0.3 else "NEGATIVE" if avg_sentiment < -0.3 else "NEUTRAL"
        trend = profile.get_recent_trend()

        print(f"{rank:2d}. {speaker:20s} │ {bar} │ {avg_sentiment:+.2f} {sentiment_label}")
        print(f"    Utterances: {len(profile.sentiment_history):3d}  │  "
              f"Primary emotion: {profile.primary_emotion.value:8s}  │  "
              f"Trend: {trend}")
        print()

    # Impact analysis
    print("\n🔍 Relationship Impact Analysis:")
    print("-" * 80)

    # Analyze "You" speaker if present
    if "You" in analytics.speakers:
        you_profile = analytics.get_speaker_profile("You")
        your_avg = you_profile.average_sentiment

        print(f"Your average sentiment: {your_avg:+.2f}")
        print(f"Your emotional volatility: {you_profile.sentiment_std:.2f}")
        print(f"Your primary emotion: {you_profile.primary_emotion.value}")
        print()

        # Find who affects you most
        print("How different people affect your sentiment:")

        speaker_impacts = []
        for speaker_id, profile in analytics.speakers.items():
            if speaker_id == "You":
                continue

            # Calculate differential
            their_avg = profile.average_sentiment
            differential = their_avg - your_avg

            speaker_impacts.append((speaker_id, differential, their_avg))

        # Sort by impact
        speaker_impacts.sort(key=lambda x: x[1])

        for speaker, diff, their_sent in speaker_impacts:
            if diff < -0.3:
                impact = "⚠️  NEGATIVE INFLUENCE"
                detail = "You tend to be more negative around this person"
            elif diff > 0.3:
                impact = "✨ POSITIVE INFLUENCE"
                detail = "You tend to be more positive around this person"
            else:
                impact = "➖ NEUTRAL"
                detail = "No significant impact on your sentiment"

            print(f"  {speaker:20s} │ {impact:20s} │ {detail}")

    # Global insights
    print("\n" + "=" * 80)
    print("💡 Key Insights:")
    print("-" * 80)

    insights = analytics.generate_insights()
    for insight in insights:
        print(f"  • {insight}")

    # Recommendations
    print("\n" + "=" * 80)
    print("🎯 Recommendations:")
    print("-" * 80)

    if "You" in analytics.speakers:
        you_profile = analytics.get_speaker_profile("You")

        if you_profile.average_sentiment < 0:
            print("  • Your overall sentiment is negative. Consider:")
            print("    - Limiting interactions with negative influences")
            print("    - Scheduling more positive activities")
            print("    - Taking breaks after stressful conversations")
        elif you_profile.average_sentiment > 0.3:
            print("  • Your overall sentiment is positive! Keep it up by:")
            print("    - Maintaining connections with positive influences")
            print("    - Continuing current healthy patterns")
        else:
            print("  • Your sentiment is neutral. To improve:")
            print("    - Seek out more positive interactions")
            print("    - Identify and reduce stress triggers")

        if you_profile.sentiment_std > 0.4:
            print("  • High emotional volatility detected. Consider:")
            print("    - Stress management techniques")
            print("    - More consistent daily routines")

    print("\n" + "=" * 80)
    print("Demo complete! This shows how Vibemaxxing tracks your vibe patterns.")
    print("=" * 80)


if __name__ == "__main__":
    simulate_day()
