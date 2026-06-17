import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("data/youtube_valence_arousal_sessions.csv")
df["datetime_baku"] = pd.to_datetime(df["datetime_baku"])
latest_day = df["datetime_baku"].dt.date.max()
df = df[df["datetime_baku"].dt.date.eq(latest_day)]
text = " ".join(df["title_clean"].dropna().astype(str).str.replace(r"#\S+", "", regex=True))

wordcloud = WordCloud(width=1600, height=900, background_color="white").generate(text)

plt.figure(figsize=(16, 9))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("visualizations/word_cloud.png", dpi=200)
