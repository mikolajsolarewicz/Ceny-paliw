"""
analysis.py
-----------
Czyste funkcje analityczne (bez zależności od Streamlit), żeby dało się je
testować i ponownie wykorzystać. Operują na ramkach pandas.

Logika merytoryczna projektu:
  1. łączenie cen detalicznych paliw z notowaniami Brent (USD/bbl) i kursem USD/PLN,
  2. przeliczenie Brent na "surowiec w PLN za litr",
  3. wyznaczenie OPÓŹNIENIA reakcji cen na stacjach (cross-correlation),
  4. pomiar ASYMETRII reakcji (efekt "rakiety i piórka"),
  5. prosta prognoza ceny na podstawie opóźnionego surowca i kursu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(__file__).parent / "data"
LITERS_PER_BARREL = 158.987  # 1 baryłka ropy = ~159 litrów
FUELS = ["PB95", "PB98", "ON"]


# --------------------------------------------------------------------------- #
#  Wczytywanie i scalanie danych
# --------------------------------------------------------------------------- #
def data_source() -> str:
    """Zwraca 'live' albo 'sample' na podstawie data/_meta.json."""
    meta_path = DATA_DIR / "_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text()).get("source", "unknown")
        except Exception:
            return "unknown"
    return "unknown"


def load_data() -> pd.DataFrame:
    """
    Wczytuje trzy pliki CSV (ceny paliw, kurs USD/PLN, Brent), scala je po dacie,
    uzupełnia dni bez notowań (weekendy/święta) metodą forward-fill i dokłada
    kolumnę 'brent_pln_l' (cena surowca w PLN za litr).
    """
    fuel = pd.read_csv(DATA_DIR / "fuel_prices.csv", parse_dates=["date"])
    usd = pd.read_csv(DATA_DIR / "usdpln.csv", parse_dates=["date"])
    brent = pd.read_csv(DATA_DIR / "brent.csv", parse_dates=["date"])

    df = fuel.merge(usd, on="date", how="outer").merge(brent, on="date", how="outer")
    df = df.sort_values("date").set_index("date")

    # Pełny kalendarz dzienny + forward-fill (kurs/Brent nie są notowane w weekendy)
    df = df.asfreq("D").ffill().dropna()

    df["brent_pln_l"] = df["brent_usd"] * df["usdpln"] / LITERS_PER_BARREL
    return df.reset_index()


# --------------------------------------------------------------------------- #
#  Opóźnienie reakcji (cross-correlation)
# --------------------------------------------------------------------------- #
def lag_correlation(df: pd.DataFrame, fuel: str, max_lag: int = 30) -> pd.DataFrame:
    """
    Korelacja dziennej zmiany ceny paliwa z dzienną zmianą ceny surowca (w PLN/l)
    opóźnioną o 0..max_lag dni. Szczyt korelacji = typowe opóźnienie reakcji stacji.
    """
    d_price = df[fuel].diff()
    d_crude = df["brent_pln_l"].diff()
    rows = []
    for lag in range(0, max_lag + 1):
        corr = d_price.corr(d_crude.shift(lag))
        rows.append({"lag": lag, "corr": corr})
    return pd.DataFrame(rows)


def best_lag(df: pd.DataFrame, fuel: str, max_lag: int = 30) -> int:
    cc = lag_correlation(df, fuel, max_lag)
    return int(cc.loc[cc["corr"].idxmax(), "lag"])


# --------------------------------------------------------------------------- #
#  Asymetria reakcji – efekt "rakiety i piórka"
# --------------------------------------------------------------------------- #
def asymmetry(df: pd.DataFrame, fuel: str, lag: int) -> dict:
    """
    Regresja dziennej zmiany ceny paliwa na osobno: WZROSTY i SPADKI opóźnionej
    ceny surowca. Jeśli współczynnik dla wzrostów > dla spadków -> stacje szybciej
    podnoszą ceny niż je obniżają ("rakieta i piórko").
    """
    d_price = df[fuel].diff()
    d_crude = df["brent_pln_l"].diff().shift(lag)

    up = d_crude.clip(lower=0)
    down = d_crude.clip(upper=0)

    data = pd.DataFrame({"y": d_price, "up": up, "down": down}).dropna()
    X = data[["up", "down"]].values
    y = data["y"].values
    model = LinearRegression().fit(X, y)
    beta_up, beta_down = model.coef_
    return {
        "beta_up": float(beta_up),      # reakcja na wzrost surowca
        "beta_down": float(beta_down),  # reakcja na spadek surowca
        "ratio": float(beta_up / beta_down) if beta_down else float("nan"),
        "asymmetric": bool(beta_up > beta_down),
    }


# --------------------------------------------------------------------------- #
#  Prognoza – regresja na opóźnionym surowcu i kursie + sezonowość
# --------------------------------------------------------------------------- #
def _seasonal_features(dates: pd.Series) -> np.ndarray:
    doy = dates.dt.dayofyear.values
    return np.column_stack([
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ])


def forecast(df: pd.DataFrame, fuel: str, lag: int, horizon: int = 30) -> pd.DataFrame:
    """
    Model: cena_paliwa ~ brent_pln_l(t-lag) + usdpln(t-lag) + sezonowość roczna.
    Prognoza to SCENARIUSZ przy założeniu, że ostatnie znane Brent i kurs się
    utrzymają (uczciwe założenie dla prostego modelu). Pas niepewności = +/-1.96*sd reszt.
    """
    work = df.copy()
    work["crude_lag"] = work["brent_pln_l"].shift(lag)
    work["usd_lag"] = work["usdpln"].shift(lag)
    work = work.dropna(subset=["crude_lag", "usd_lag", fuel])

    seas = _seasonal_features(work["date"])
    X = np.column_stack([work["crude_lag"].values, work["usd_lag"].values, seas])
    y = work[fuel].values
    model = LinearRegression().fit(X, y)

    fitted = model.predict(X)
    resid_sd = float(np.std(y - fitted, ddof=1))

    # Scenariusz: ostatnie znane Brent/kurs utrzymane na przyszłość
    last_crude = float(df["brent_pln_l"].iloc[-1])
    last_usd = float(df["usdpln"].iloc[-1])
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    fseas = _seasonal_features(pd.Series(future_dates))
    Xf = np.column_stack([
        np.full(horizon, last_crude),
        np.full(horizon, last_usd),
        fseas,
    ])
    yhat = model.predict(Xf)

    hist = pd.DataFrame({"date": work["date"].values, "value": y, "kind": "historia"})
    fit = pd.DataFrame({"date": work["date"].values, "value": fitted, "kind": "dopasowanie"})
    fut = pd.DataFrame({
        "date": future_dates,
        "value": yhat,
        "lo": yhat - 1.96 * resid_sd,
        "hi": yhat + 1.96 * resid_sd,
        "kind": "prognoza",
    })
    out = pd.concat([hist, fit, fut], ignore_index=True)
    out.attrs["r2"] = float(model.score(X, y))
    out.attrs["resid_sd"] = resid_sd
    return out


if __name__ == "__main__":
    # Szybki test logiki na danych z katalogu data/
    df = load_data()
    print(f"Źródło danych: {data_source()}")
    print(f"Zakres: {df['date'].min().date()} – {df['date'].max().date()}  ({len(df)} dni)")
    for f in FUELS:
        lag = best_lag(df, f)
        asym = asymmetry(df, f, lag)
        print(f"  {f}: opóźnienie={lag} dni | β_wzrost={asym['beta_up']:.3f} "
              f"β_spadek={asym['beta_down']:.3f} asymetria={asym['asymmetric']}")
    fc = forecast(df, "PB95", best_lag(df, "PB95"), horizon=30)
    print(f"  Prognoza PB95: R²={fc.attrs['r2']:.3f}, "
          f"ostatnia prognoza={fc[fc.kind=='prognoza']['value'].iloc[-1]:.2f} zł/l")
