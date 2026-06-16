import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("youtube_valence_arousal_sessions.csv")
df = df.sort_values(["session_id", "session_step"])
df["datetime_baku"] = pd.to_datetime(df["datetime_baku"])
df["step_size"] = (
    df.groupby("session_id")[["valence", "arousal"]]
    .diff()
    .pow(2)
    .sum(axis=1)
    .pow(0.5)
    .fillna(0)
)

sessions = list(df.groupby("session_id"))
fig, axes = plt.subplots(len(sessions), 1, figsize=(10, 3 * len(sessions)), sharex=False)
if len(sessions) == 1:
    axes = [axes]

def format_duration(delta):
    minutes = int(delta.total_seconds() // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"

for ax, (session_id, session) in zip(axes, sessions):
    duration = format_duration(session["datetime_baku"].iloc[-1] - session["datetime_baku"].iloc[0])
    for i in range(1, len(session)):
        color = "green" if session["valence"].iloc[i] >= 0 else "red"
        ax.plot(
            session["session_step"].iloc[i - 1:i + 1],
            session["valence"].iloc[i - 1:i + 1],
            marker="o",
            color=color,
            label="valence" if i == 1 else None,
        )
    ax.plot(session["session_step"], session["arousal"], marker="o", label="arousal")
    ax.set_title(f"{session_id} ({duration})")
    ax.set_ylim(-1, 1)
    ax.set_xticks(session["session_step"])
    ax.set_xlabel("Session step")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)

fig.suptitle("Session Mood Trajectories", y=0.995)
plt.tight_layout()
plt.savefig("session_trajectories.png", dpi=200)
