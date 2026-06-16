import json
import os
import pandas as pd
from zoneinfo import ZoneInfo
from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

INPUT_FILE = "watch-history.json"
OUTPUT_FILE = "youtube_valence_arousal_sessions.csv"
SELECTED_INDICES = range(0, 50)

ValenceArousalClass = Literal[
    "negative_low_arousal",
    "negative_moderate_arousal",
    "negative_high_arousal",
    "neutral_low_arousal",
    "neutral_moderate_arousal",
    "neutral_high_arousal",
    "positive_low_arousal",
    "positive_moderate_arousal",
    "positive_high_arousal",
]

load_dotenv(r"D:\youtube\code\.env")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

class Score(BaseModel):
    item_id: int
    valence_arousal_class: ValenceArousalClass
    valence: float = Field(ge=-1, le=1)
    arousal: float = Field(ge=-1, le=1)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame([data[i] for i in SELECTED_INDICES])
df = df[~df["title"].str.strip().str.startswith("Viewed a post", na=False)].copy()
df = df[~df["titleUrl"].str.contains("/post/", na=False)].copy()
df["title_clean"] = df["title"].str.replace(r"^(Watched|Viewed)\s+", "", regex=True)

df["datetime_baku_dt"] = (
    pd.to_datetime(df["time"], utc=True, format="ISO8601")
    .dt.tz_convert(ZoneInfo("Asia/Baku"))
    .dt.tz_localize(None)
)

df = df.sort_values("datetime_baku_dt").reset_index(drop=True)
df["item_id"] = range(len(df))
df["session_id"] = (df["datetime_baku_dt"].diff().gt(pd.Timedelta(hours=1)).cumsum() + 1).map(lambda x: f"S{x:03d}")
df["session_step"] = df.groupby("session_id").cumcount() + 1
df["datetime_baku"] = df["datetime_baku_dt"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
    response = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "system",
                "content": (
                    "Classify the title in the valence-arousal model. "
                    "Valence and arousal must be floats from -1 to 1 centered at 0. "
                    "The valence_arousal_class must be exactly one of: "
                    "negative_low_arousal, negative_moderate_arousal, negative_high_arousal, "
                    "neutral_low_arousal, neutral_moderate_arousal, neutral_high_arousal, "
                    "positive_low_arousal, positive_moderate_arousal, positive_high_arousal. "
                    "Return the same item_id value."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"item_id": row["item_id"], "title_clean": row["title_clean"]}, ensure_ascii=False),
            },
        ],
        text_format=Score,
    )

    score = response.output_parsed
    valence = round(score.valence, 1)
    arousal = round(score.arousal, 1)

    pd.DataFrame([{
        "session_id": row["session_id"],
        "session_step": row["session_step"],
        "datetime_baku": row["datetime_baku"],
        "title_clean": row["title_clean"],
        "valence_arousal_class": score.valence_arousal_class,
        "valence": valence,
        "arousal": arousal,
    }]).to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not os.path.exists(OUTPUT_FILE),
        index=False,
        encoding="utf-8-sig",
    )
