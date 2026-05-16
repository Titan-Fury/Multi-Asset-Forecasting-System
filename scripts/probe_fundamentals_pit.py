"""Verify fundamentals PIT snapshot save/load round-trip."""
import pandas as pd
from omni.storage.fundamentals_snapshots import (
    snapshot_save, load_pit, list_snapshots,
)

ac, tk = "india", "_TEST_PIT"

# Day 1 snapshot
df1 = pd.DataFrame([{"pe": 15.0, "roe": 0.12, "revenue_growth": 0.08}])
snapshot_save(ac, tk, df1, snapshot_date=pd.Timestamp("2026-04-01"))

# Day 2 snapshot
df2 = pd.DataFrame([{"pe": 18.5, "roe": 0.14, "revenue_growth": 0.09}])
snapshot_save(ac, tk, df2, snapshot_date=pd.Timestamp("2026-05-08"))

# Day 3 snapshot
df3 = pd.DataFrame([{"pe": 22.0, "roe": 0.13, "revenue_growth": 0.11}])
snapshot_save(ac, tk, df3, snapshot_date=pd.Timestamp("2026-05-15"))

print("All snapshots:")
print(list_snapshots(ac, tk))
print()
print("PIT @ 2026-04-15 (should pick April 1):")
print(load_pit(ac, tk, "2026-04-15"))
print()
print("PIT @ 2026-05-10 (should pick May 8):")
print(load_pit(ac, tk, "2026-05-10"))
print()
print("PIT @ 2026-03-01 (no snapshot before):")
r = load_pit(ac, tk, "2026-03-01")
print(f"empty: {r.empty}")
print()
print("PIT @ None (latest):")
print(load_pit(ac, tk, None))

# Idempotency check: re-write May 8 with new value
print()
print("Re-write May 8 with pe=999 (idempotent: should replace, not append)...")
snapshot_save(ac, tk, pd.DataFrame([{"pe": 999.0, "roe": 0.50}]),
               snapshot_date=pd.Timestamp("2026-05-08"))
print(list_snapshots(ac, tk))

# Cleanup
from pathlib import Path
from omni.utils.paths import local_data_root
p = local_data_root() / "lake" / "fundamentals_pit" / ac / "_TEST_PIT.parquet"
if p.exists():
    p.unlink()
print(f"cleaned up {p}")
