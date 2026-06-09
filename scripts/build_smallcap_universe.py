import re
import pandas as pd

# Mega-cap / F&O names to exclude (already covered as large-cap large-liquidity)
MEGA = {
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
    "ITC","LT","KOTAKBANK","AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA","TITAN",
    "BAJFINANCE","WIPRO","HCLTECH","ULTRACEMCO","ADANIENT","ADANIPORTS","ONGC","NTPC",
    "POWERGRID","COALINDIA","TATAMOTORS","TATASTEEL","JSWSTEEL","HINDALCO","BAJAJFINSV",
    "HDFCLIFE","SBILIFE","ICICIPRULI","DIVISLAB","DRREDDY","CIPLA","APOLLOHOSP",
    "BRITANNIA","NESTLEIND","DABUR","GODREJCP","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT",
    "M&M","GRASIM","SHRIRAMFIN","INDUSINDBK","BPCL","IOC","TECHM","LTIM","DMART",
    "JIOFIN","LICI","VBL","DLF","ADANIGREEN","ADANIPOWER","BANKBARODA","PIDILITIND",
    "PNB","ZOMATO","IRFC","INDIGO","HAL","BEL",
}

def is_etf_or_fund(sym: str) -> bool:
    s = sym.upper()
    return any(tok in s for tok in ("BEES", "ETF", "LIQUID", "GOLD", "NIFTY", "SENSEX",
                                     "SETF", "IETF", "MOM", "MAFANG", "MNC"))

df = pd.read_csv("NSE_bhavcopy/sec_bhavdata_full_19062026.csv", skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
eq = df[df["SERIES"] == "EQ"].copy()
eq["TURNOVER_LACS"] = pd.to_numeric(eq["TURNOVER_LACS"], errors="coerce")
eq = eq.dropna(subset=["TURNOVER_LACS"]).sort_values("TURNOVER_LACS", ascending=False).reset_index(drop=True)

RANK_LO, RANK_HI = 55, 305   # mid/small-cap liquid band
band = eq.loc[RANK_LO:RANK_HI]
uni = [s for s in band["SYMBOL"].tolist()
       if s not in MEGA and not is_etf_or_fund(s) and re.match(r"^[A-Z0-9&-]+$", s)]
print("band size:", len(band), "-> universe after filters:", len(uni))
print("\nfirst 40:", uni[:40])
print("\nlast 40:", uni[-40:])

# write config for reproducibility
with open("configs/universe_smallcap.yaml", "w") as f:
    f.write("# NSE mid/small-cap universe (data-derived)\n")
    f.write("# Source: NSE sec_bhavdata_full 2026-06-19, EQ series, turnover rank "
            f"{RANK_LO}-{RANK_HI}, excl. mega-cap F&O + ETFs.\n")
    f.write(f"# Count: {len(uni)}\n")
    f.write("asset_class: india\n")
    f.write("tickers:\n")
    for t in uni:
        f.write(f"  - {t}\n")
print("\nwrote configs/universe_smallcap.yaml")
