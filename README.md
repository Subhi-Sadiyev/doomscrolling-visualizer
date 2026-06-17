# YouTube Mood Analysis

Minimal pipeline for analyzing YouTube watch history with the valence-arousal model.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

- `pre_processing.py`: reads `data/watch-history.json` (get it from https://takeout.google.com/), fetches video details, classifies each title, and writes `data/youtube_valence_arousal_sessions.csv`
- `line_chart.py`: renders session mood line charts
- `directed_graph.py`: renders session mood direction plots
- `word_cloud.py`: renders a word cloud for sessions
- `get_video_details.py`: helper for fetching video description and top comments

## Output

Generated plots are written to `visualizations/`.
