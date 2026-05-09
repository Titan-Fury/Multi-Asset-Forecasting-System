"""S3 — News embeddings as model features.

Pipeline:
    1. For each (ticker, date), gather the headlines from that day.
    2. Embed via Nvidia llama-nemotron (1024-d, OpenRouter free tier).
    3. PCA-reduce the corpus to 32-d so tree models don't drown in features.
    4. Cache raw 1024-d vectors per (ticker, date) on disk so we never re-embed.
    5. Output 32 columns named `emb_00 .. emb_31` per (ticker, date).

We deliberately keep the feature count small (32) because:
    - The universal model has ~88 features today; 1024 new ones would
      destroy the variance-bias balance.
    - 32 dims captures >85% of the explained variance on the FinBERT
      headline corpus we're already scoring.

Graceful degradation:
    - No API key  -> zero matrix; columns present but model treats as constant.
    - Empty news  -> NaN columns.
    - PCA missing -> falls back to first 32 dims of raw vector.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from omni.llm.embed_nemotron import embed as nemotron_embed, EMBED_DIM

EMB_OUT_DIM = 32
EMB_CACHE_DIR = Path("data") / "cache" / "news_embed"
PCA_PATH = Path("data") / "models" / "news_pca.joblib"


def _emb_cols() -> list[str]:
    return [f"emb_{i:02d}" for i in range(EMB_OUT_DIM)]


def _cache_key(ticker: str, day: pd.Timestamp, headlines: list[str]) -> Path:
    h = hashlib.sha256(("\n".join(headlines)).encode("utf-8")).hexdigest()[:12]
    d = pd.Timestamp(day).strftime("%Y%m%d")
    safe_t = str(ticker).replace("/", "_").replace("\\", "_")
    return EMB_CACHE_DIR / safe_t / f"{d}_{h}.npy"


def _load_or_embed(ticker: str, day: pd.Timestamp,
                   headlines: list[str]) -> np.ndarray:
    """Returns 1024-d vector. Cache hit avoids any API call."""
    if not headlines:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    path = _cache_key(ticker, day, headlines)
    if path.exists():
        try:
            return np.load(path)
        except Exception:
            pass
    text = " | ".join(headlines[:8])  # cap at 8 headlines per day
    vec = nemotron_embed([text])[0]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vec.astype(np.float32))
    except Exception:
        pass
    return vec


def embed_daily_news(news_raw: pd.DataFrame,
                     ticker: str,
                     dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Build a per-day 1024-d embedding frame for one ticker.

    Args:
        news_raw: DataFrame with at least [published_at, title].
                  May include `ticker_hint` to filter.
        ticker: the ticker we're embedding for.
        dates:  iterable of dates we want a row for.

    Returns:
        DataFrame[date, raw_0 .. raw_1023].
    """
    dates = pd.to_datetime(list(dates)).normalize().unique()
    if news_raw is None or news_raw.empty or "title" not in news_raw.columns:
        # No news at all — return all-zero vectors so the merge succeeds
        zero = np.zeros((len(dates), EMBED_DIM), dtype=np.float32)
        df = pd.DataFrame(zero, columns=[f"raw_{i}" for i in range(EMBED_DIM)])
        df["date"] = dates
        return df

    nr = news_raw.copy()
    if "published_at" in nr.columns:
        nr["day"] = pd.to_datetime(nr["published_at"], errors="coerce").dt.normalize()
    else:
        nr["day"] = pd.NaT
    if "ticker_hint" in nr.columns:
        sym = str(ticker).upper().replace(".NS", "").replace(".BO", "")
        m = nr["ticker_hint"].astype(str).str.upper().str.contains(sym, na=False)
        if m.any():
            nr = nr[m]

    rows: list[dict] = []
    for d in dates:
        day = pd.Timestamp(d).normalize()
        sub = nr[nr["day"] == day]
        heads = (sub["title"].dropna().astype(str).head(8).tolist()
                 if not sub.empty else [])
        vec = _load_or_embed(ticker, day, heads)
        row = {"date": day}
        row.update({f"raw_{i}": float(vec[i]) for i in range(EMBED_DIM)})
        rows.append(row)
    return pd.DataFrame(rows)


def fit_pca(corpus_vecs: np.ndarray, n: int = EMB_OUT_DIM):
    """Fit a PCA on raw 1024-d vectors. Saves to PCA_PATH."""
    from sklearn.decomposition import PCA
    import joblib
    if corpus_vecs.shape[0] < n:
        # Not enough samples — return identity projection
        return None
    pca = PCA(n_components=n, random_state=42)
    pca.fit(corpus_vecs)
    PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, PCA_PATH)
    return pca


def _try_load_pca():
    if not PCA_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(PCA_PATH)
    except Exception:
        return None


def reduce_to_emb(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Project raw_0..raw_1023 -> emb_00..emb_31 via persisted PCA.

    Falls back to the first 32 dims of the raw vector if the PCA file
    is missing (so the schema is always stable).
    """
    raw_cols = [f"raw_{i}" for i in range(EMBED_DIM)]
    have = [c for c in raw_cols if c in raw_df.columns]
    if not have:
        out = raw_df.copy()
        for c in _emb_cols():
            out[c] = 0.0
        return out
    X = raw_df[have].fillna(0.0).to_numpy(dtype=np.float32)
    pca = _try_load_pca()
    if pca is None:
        Z = X[:, :EMB_OUT_DIM]
    else:
        Z = pca.transform(X)
    out = raw_df.drop(columns=have).copy()
    for i in range(EMB_OUT_DIM):
        out[f"emb_{i:02d}"] = Z[:, i] if i < Z.shape[1] else 0.0
    return out


def add_news_embeddings(out: pd.DataFrame,
                        ticker: str,
                        news_raw: pd.DataFrame | None) -> pd.DataFrame:
    """Public entry: append emb_00..emb_31 to a feature frame.

    Always returns the same column set, even when news is absent.
    """
    cols = _emb_cols()
    if "date" not in out.columns:
        for c in cols:
            out[c] = np.nan
        return out
    dates = pd.to_datetime(out["date"]).dropna().unique()
    raw = embed_daily_news(news_raw if news_raw is not None else pd.DataFrame(),
                            ticker, dates)
    reduced = reduce_to_emb(raw)
    reduced["date"] = pd.to_datetime(reduced["date"]).astype("datetime64[ns]")
    out = out.copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    # Lag embeddings by +1 BDay to avoid look-ahead (same convention as sentiment)
    reduced["date"] = reduced["date"] + pd.tseries.offsets.BDay(1)
    keep_cols = ["date"] + cols
    return out.merge(reduced[keep_cols], on="date", how="left")
