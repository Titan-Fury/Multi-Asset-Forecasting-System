import re, json, time, random, urllib.request, urllib.error

env = {}
for l in open(".env"):
    if re.match(r"^[A-Za-z]", l):
        k, _, v = l.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
PKEY = env["POLYGON_API_KEY"]


def page(url):
    for _ in range(6):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(13); continue
            raise
    return {}


def collect(active, max_pages):
    out, url = [], (f"https://api.polygon.io/v3/reference/tickers?market=stocks&type=CS"
                    f"&active={active}&limit=1000&apiKey={PKEY}")
    for _ in range(max_pages):
        b = page(url)
        for s in b.get("results", []):
            t = s.get("ticker", "")
            # plain US common-stock symbols only (skip warrants/units/dots)
            if re.match(r"^[A-Z]{1,5}$", t):
                out.append((t, s.get("delisted_utc")))
        nxt = b.get("next_url")
        if not nxt:
            break
        url = nxt + f"&apiKey={PKEY}"
        time.sleep(13)  # 5/min free tier
    return out


active = collect("true", 4)
delisted = collect("false", 4)
print(f"polygon CS: active={len(active)} delisted={len(delisted)}")

random.seed(7)
# survivorship-clean sample: deliberately include many delisted losers
samp_active = random.sample(active, min(150, len(active)))
samp_delisted = random.sample(delisted, min(150, len(delisted)))
uni = sorted(set(t for t, _ in samp_active) | set(t for t, _ in samp_delisted))
delisted_set = {t for t, _ in delisted}
n_del = sum(1 for t in uni if t in delisted_set)
print(f"sampled universe: {len(uni)} ({n_del} delisted, {len(uni)-n_del} active)")

with open("configs/universe_us_pit.yaml", "w") as f:
    f.write("# US survivorship-clean common-stock sample (incl. delisted)\n")
    f.write("# Source: Polygon /v3/reference/tickers type=CS, active true+false, 2026-06.\n")
    f.write("# 'small/illiquid' is defined PIT in-eval via daily dollar-volume cut, not here.\n")
    f.write(f"# Count: {len(uni)} ({n_del} delisted)\n")
    f.write("source: tiingo+yfinance\n")
    f.write("tickers:\n")
    for t in uni:
        f.write(f"  - {t}\n")
    f.write("delisted:  # routed to Tiingo (yfinance lacks delisted names)\n")
    for t in sorted(t for t in uni if t in delisted_set):
        f.write(f"  - {t}\n")
print("wrote configs/universe_us_pit.yaml")
