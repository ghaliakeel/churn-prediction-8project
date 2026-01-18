"""
Data loading and preprocessing.
"""
import json
from pathlib import Path

import pandas as pd


def load_events(filepath: str | Path) -> pd.DataFrame:
    """Load event logs from JSON lines file."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["registration"] = pd.to_datetime(df["registration"], unit="ms")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # drop rows with missing/empty userId
    df = df[df["userId"].notna() & (df["userId"] != "")]
    df["userId"] = df["userId"].astype(str)
    return df.copy()


def identify_churned_users(df: pd.DataFrame) -> set:
    # churn = user reached "Cancellation Confirmation" page
    churned = df[df["page"] == "Cancellation Confirmation"]["userId"].unique()
    return set(churned)


def create_user_features(df: pd.DataFrame, churned_users: set) -> pd.DataFrame:
    """
    Convert event logs to user-level features.
    Only use data BEFORE churn to avoid leakage.
    """
    # get churn timestamps
    churn_times = (
        df[df["page"] == "Cancellation Confirmation"]
        .groupby("userId")["ts"]
        .min()
        .to_dict()
    )

    # filter out post-churn events
    def is_pre_churn(row):
        uid = row["userId"]
        if uid in churn_times:
            return row["ts"] < churn_times[uid]
        return True

    df_filtered = df[df.apply(is_pre_churn, axis=1)].copy()

    features = []

    for uid, udf in df_filtered.groupby("userId"):
        n_sessions = udf["sessionId"].nunique()
        n_songs = len(udf[udf["page"] == "NextSong"])
        n_thumbs_up = len(udf[udf["page"] == "Thumbs Up"])
        n_thumbs_down = len(udf[udf["page"] == "Thumbs Down"])
        n_playlist = len(udf[udf["page"] == "Add to Playlist"])
        n_friends = len(udf[udf["page"] == "Add Friend"])
        n_errors = len(udf[udf["page"] == "Error"])
        n_help = len(udf[udf["page"] == "Help"])
        n_downgrade = len(udf[udf["page"] == "Downgrade"])
        n_ads = len(udf[udf["page"] == "Roll Advert"])

        udf_sorted = udf.sort_values("ts")
        first_ts = udf_sorted["ts"].min()
        last_ts = udf_sorted["ts"].max()
        days_active = (last_ts - first_ts).days + 1

        listen_time = udf[udf["page"] == "NextSong"]["length"].sum()

        songs_per_sess = n_songs / max(n_sessions, 1)
        thumbs_ratio = n_thumbs_up / max(n_thumbs_up + n_thumbs_down, 1)

        level = udf_sorted["level"].iloc[-1] if len(udf_sorted) > 0 else "free"
        is_paid = 1 if level == "paid" else 0

        gender = udf["gender"].iloc[0] if len(udf) > 0 else "U"
        is_male = 1 if gender == "M" else 0

        features.append({
            "userId": uid,
            "n_sessions": n_sessions,
            "n_songs": n_songs,
            "n_thumbs_up": n_thumbs_up,
            "n_thumbs_down": n_thumbs_down,
            "n_add_playlist": n_playlist,
            "n_add_friend": n_friends,
            "n_errors": n_errors,
            "n_help": n_help,
            "n_downgrade": n_downgrade,
            "n_adverts": n_ads,
            "days_active": days_active,
            "total_listen_time": listen_time,
            "songs_per_session": songs_per_sess,
            "thumbs_ratio": thumbs_ratio,
            "is_paid": is_paid,
            "is_male": is_male,
            "churned": 1 if uid in churned_users else 0,
        })

    return pd.DataFrame(features)
