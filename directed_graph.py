import pandas as pd
import matplotlib
from zoneinfo import ZoneInfo
from datetime import datetime

matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/youtube_valence_arousal_sessions.csv")
df = df.sort_values(["session_id", "session_step"])
df["datetime_baku"] = pd.to_datetime(df["datetime_baku"])
today = datetime.now(ZoneInfo("Asia/Baku")).date()
target_date = today if df["datetime_baku"].dt.date.eq(today).any() else df["datetime_baku"].dt.date.max()

sessions = [
    (session_id, session)
    for session_id, session in df.groupby("session_id")
    if session["datetime_baku"].dt.date.eq(target_date).any()
]
fig, axes = plt.subplots(len(sessions), 1, figsize=(8, 3 * len(sessions)))
if len(sessions) == 1:
    axes = [axes]

def format_duration(delta):
    minutes = int(delta.total_seconds() // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"

for ax, (session_id, session) in zip(axes, sessions):
    start_time = session["datetime_baku"].iloc[0].strftime("%H:%M")
    end_time = session["datetime_baku"].iloc[-1].strftime("%H:%M")
    length = format_duration(session["datetime_baku"].iloc[-1] - session["datetime_baku"].iloc[0])
    thirds = [session.iloc[: len(session) // 3], session.iloc[len(session) // 3 : 2 * len(session) // 3], session.iloc[2 * len(session) // 3 :]]
    points = pd.DataFrame([
        chunk[["valence", "arousal"]].mean() for chunk in thirds if len(chunk)
    ]).to_numpy()
    labels = ["start", "middle", "end"]
    ax.scatter(points[:, 0], points[:, 1], s=60)
    for i in range(len(points) - 1):
        ax.annotate(
            "",
            xy=(points[i + 1, 0], points[i + 1, 1]),
            xytext=(points[i, 0], points[i, 1]),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
        )
    for (x, y), label in zip(points, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_title(session_id, pad=22)
    ax.text(
        0.5,
        1.08,
        f"Start: {start_time}   End: {end_time}   Duration: {length}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    pad = 0.1
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)

axes[-1].set_xlabel("Valence")
for ax in axes:
    ax.set_ylabel("Arousal")
fig.suptitle(f"Mood Direction for {target_date.isoformat()}", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("visualizations/directed_graph_visualization.png", dpi=200)
