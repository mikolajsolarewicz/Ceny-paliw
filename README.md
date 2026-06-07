"""
fetch_data.py
-------------
Pobiera PRAWDZIWE dane i zapisuje do katalogu data/ jako CSV:
  - ceny detaliczne paliw (PB95/PB98/ON)  -> api.paliwo.today
  - kurs USD/PLN                          -> api.nbp.pl (NBP, tabela A)
  - notowania ropy Brent (USD/bbl)        -> yfinance (ticker BZ=F)

Uruchom RAZ przed publikacją dashboardu:
    python fetch_data.py
a następnie zacommituj zawartość data/ do repo (dane będą gotowe na deployu).

Uwaga: ten skrypt wymaga dostępu do internetu. Jeśli któreś źródło zawiedzie,
skrypt poinformuje o tym, a istniejące dane (np. przykładowe) nie zostaną nadpisane.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
START = "2020-01-01"
FUELS = ["PB95", "PB98", "ON"]


def fetch_fuel_prices() -> pd.DataFrame:
    """Ceny detaliczne z publicznego API paliwo.today (jeden request na rodzaj paliwa)."""
    frames = []
    for fuel in FUELS:
        url = f"https://api.paliwo.today/api/prices?type={fuel}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        rec = pd.DataFrame(r.json())
        rec["date"] = pd.to_datetime(rec["date"]).dt.tz_localize(None).dt.normalize()
        rec["price"] = pd.to_numeric(rec["price"], errors="coerce")
        rec = rec[["date", "price"]].rename(columns={"price": fuel})
        frames.append(rec.set_index("date"))
    df = pd.concat(frames, axis=1).sort_index().reset_index()
    return df[df["date"] >= pd.Timestamp(START)]


def fetch_usdpln() -> pd.DataFrame:
    """Kurs średni USD/PLN z API NBP (tabela A). NBP ogranicza zapytanie do 367 dni."""
    out = []
    start = date.fromisoformat(START)
    today = date.today()
    while start < today:
        end = min(start + timedelta(days=366), today)
        url = (f"http://api.nbp.pl/api/exchangerates/rates/a/usd/"
               f"{start.isoformat()}/{end.isoformat()}/?format=json")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            for row in r.json()["rates"]:
                out.append({"date": row["effectiveDate"], "usdpln": row["mid"]})
        start = end + timedelta(days=1)
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").sort_values("date")


def fetch_brent() -> pd.DataFrame:
    """Notowania Brent (BZ=F) z yfinance."""
    import yfinance as yf
    data = yf.download("BZ=F", start=START, progress=False, auto_adjust=True)
    close = data["Close"]
    if hasattr(close, "columns"):       # MultiIndex w nowszych wersjach yfinance
        close = close.iloc[:, 0]
    df = close.reset_index()
    df.columns = ["date", "brent_usd"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.dropna()


def main() -> None:
    ok = True
    try:
        print("→ Pobieram ceny paliw (paliwo.today)…")
        fuel = fetch_fuel_prices()
        print(f"   {len(fuel)} rekordów, do {fuel['date'].max().date()}")
        print("→ Pobieram kurs USD/PLN (NBP)…")
        usd = fetch_usdpln()
        print(f"   {len(usd)} rekordów, do {usd['date'].max().date()}")
        print("→ Pobieram Brent (yfinance)…")
        brent = fetch_brent()
        print(f"   {len(brent)} rekordów, do {brent['date'].max().date()}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n‼  Błąd pobierania: {exc}")
        print("   Dane NIE zostały nadpisane. Sprawdź połączenie i spróbuj ponownie.")
        ok = False

    if ok:
        fuel.to_csv(DATA_DIR / "fuel_prices.csv", index=False)
        usd.to_csv(DATA_DIR / "usdpln.csv", index=False)
        brent.to_csv(DATA_DIR / "brent.csv", index=False)
        (DATA_DIR / "_meta.json").write_text(json.dumps({
            "source": "live",
            "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }, indent=2))
        print("\n✓ Zapisano prawdziwe dane do data/. Możesz commitować i deployować.")


if __name__ == "__main__":
    main()
