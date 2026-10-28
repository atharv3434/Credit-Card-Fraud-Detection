"""Simulate 90 days of card transactions for 4,000 users with three realistic fraud patterns.

  card_testing      bursts of tiny online purchases at new devices, sometimes ending in a large cash-out
  account_takeover  new device, foreign IP, spending 3-8x the user's norm on electronics/gift cards/travel
                    (20% are "hard": same device, home country - only the amount gives them away)
  stolen_card       in-person foreign purchases of expensive goods (20% "hard": home country)

Legitimate noise is deliberate so the problem is not trivially separable: ~3% of legitimate
transactions are abroad (travel), ~1% are unusually large purchases.
fraud_type is saved for error analysis ONLY and is never used as a feature.
"""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

rng = np.random.default_rng(C.SEED)
N_USERS, DAYS, N_MERCHANTS = 4000, 90, 800
COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "NG", "RU"]
CATS = ["grocery", "restaurant", "fuel", "electronics", "travel", "digital_goods", "gift_cards", "jewelry", "clothing", "utilities"]
CAT_P = np.array([.26, .20, .12, .05, .04, .08, .02, .02, .11, .10]); CAT_P /= CAT_P.sum()
CAT_MU = dict(zip(CATS, [3.4, 3.2, 3.7, 5.3, 5.5, 2.8, 3.9, 5.6, 4.1, 4.3]))
CAT_ONLINE = dict(zip(CATS, [.1, .1, 0.0, .55, .85, .95, .8, .3, .4, .7]))
merch_cat = rng.choice(len(CATS), N_MERCHANTS, p=CAT_P)
merch_by_cat = {c: np.where(merch_cat == i)[0] + 1 for i, c in enumerate(CATS)}
HOUR_W = np.array([1, .5, .4, .3, .3, .6, 1.5, 3, 4, 4.5, 5, 5.5, 6, 5.5, 5, 5, 5.5, 6, 6, 5.5, 4.5, 3.5, 2.5, 1.5]); HOUR_W = HOUR_W / HOUR_W.sum()

users = pd.DataFrame({
    "user_id": np.arange(1, N_USERS + 1),
    "home_country": rng.choice(COUNTRIES, N_USERS, p=[.70, .06, .06, .05, .05, .03, .03, .02]),
    "account_age_days": (rng.exponential(600, N_USERS) + 5).astype(int)})
scale = rng.lognormal(0, 0.4, N_USERS)
rate = rng.lognormal(np.log(0.7), 0.5, N_USERS)

rows = []   # (user, ts, amount, merchant, category, channel, country, device, is_fraud, fraud_type)


def rand_ts(day_lo=0, day_hi=DAYS):
    day = int(rng.integers(day_lo, day_hi)); hour = int(rng.choice(24, p=HOUR_W))
    return C.START + timedelta(days=day, hours=hour, minutes=int(rng.integers(60)), seconds=int(rng.integers(60)))


def merchant(cat):
    return int(rng.choice(merch_by_cat[cat]))


for u in range(N_USERS):
    uid, home = u + 1, users.home_country[u]
    for _ in range(rng.poisson(rate[u] * DAYS)):
        cat = CATS[rng.choice(len(CATS), p=CAT_P)]
        amt = float(np.exp(rng.normal(CAT_MU[cat] + np.log(scale[u]), 0.55)))
        if rng.random() < 0.01: amt *= rng.uniform(3, 6)                      # legit big purchase
        online = rng.random() < CAT_ONLINE[cat]
        ctry = home if rng.random() < 0.97 else str(rng.choice([c for c in COUNTRIES if c != home]))
        dev = (f"D{uid}-{0 if rng.random() < .9 else 1}") if online else ""
        rows.append((uid, rand_ts(), round(amt, 2), merchant(cat), cat, "online" if online else "pos", ctry, dev, 0, ""))


def foreign(home):
    return str(rng.choice([c for c in COUNTRIES if c != home]))


def typical(u):
    return float(np.exp(3.9 + np.log(scale[u - 1])))


for _ in range(70):                                                     # --- card testing
    u = int(rng.integers(1, N_USERS + 1)); home = users.home_country[u - 1]
    t = rand_ts(5, DAYS); dev = f"ATK-{rng.integers(1e6)}"; ctry = foreign(home) if rng.random() < .5 else home
    for _ in range(int(rng.integers(6, 17))):
        t += timedelta(seconds=int(rng.integers(10, 120)))
        rows.append((u, t, round(float(rng.uniform(.5, 6)), 2), merchant("digital_goods"), "digital_goods", "online", ctry, dev, 1, "card_testing"))
    if rng.random() < .5:
        t += timedelta(seconds=int(rng.integers(30, 300))); cat = str(rng.choice(["electronics", "gift_cards"]))
        rows.append((u, t, round(float(rng.uniform(150, 600)), 2), merchant(cat), cat, "online", ctry, dev, 1, "card_testing"))

for _ in range(330):                                                    # --- account takeover
    u = int(rng.integers(1, N_USERS + 1)); home = users.home_country[u - 1]; hard = rng.random() < .2
    t = rand_ts(5, DAYS); dev = f"D{u}-0" if hard else f"ATK-{rng.integers(1e6)}"
    ctry = home if hard or rng.random() < .45 else foreign(home)
    for _ in range(int(rng.integers(1, 5))):
        t += timedelta(minutes=int(rng.integers(1, 30))); cat = str(rng.choice(["electronics", "gift_cards", "travel"]))
        rows.append((u, t, round(typical(u) * float(rng.uniform(3, 8)), 2), merchant(cat), cat, "online", ctry, dev, 1, "account_takeover"))

for _ in range(180):                                                    # --- stolen card, in person
    u = int(rng.integers(1, N_USERS + 1)); home = users.home_country[u - 1]; hard = rng.random() < .2
    t = rand_ts(5, DAYS); ctry = home if hard else foreign(home)
    for _ in range(int(rng.integers(1, 4))):
        t += timedelta(minutes=int(rng.integers(5, 40))); cat = str(rng.choice(["electronics", "jewelry", "clothing"]))
        rows.append((u, t, round(float(rng.uniform(150, 900)), 2), merchant(cat), cat, "pos", ctry, "", 1, "stolen_card"))

tx = pd.DataFrame(rows, columns=["user_id", "ts", "amount", "merchant_id", "merchant_category", "channel", "country", "device_id", "is_fraud", "fraud_type"])
tx = tx[tx.ts < C.START + timedelta(days=DAYS)].sort_values(["ts", "user_id"]).reset_index(drop=True)
tx.insert(0, "txn_id", [f"T{i:07d}" for i in range(1, len(tx) + 1)])
tx["ts"] = tx.ts.dt.strftime("%Y-%m-%d %H:%M:%S")
C.DATA.mkdir(exist_ok=True)
tx.to_csv(C.TX_CSV, index=False); users.to_csv(C.USERS_CSV, index=False)
print(f"transactions={len(tx):,} users={N_USERS:,} fraud={tx.is_fraud.sum():,} ({tx.is_fraud.mean():.2%})")
print(tx[tx.is_fraud == 1].fraud_type.value_counts().to_dict(), f"| fraud $ = {tx[tx.is_fraud == 1].amount.sum():,.0f}")
