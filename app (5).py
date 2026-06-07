"""
Analiza cen paliw w Polsce (2020-2026) - dashboard Streamlit.

Wersja jednoplikowa i samowystarczalna: zawiera analize, generator danych
oraz interfejs. Jesli w katalogu ./data sa poprawne pliki CSV (np. z fetch_data.py),
uzywa ich; w przeciwnym razie generuje realistyczne dane przykladowe w pamieci.

Uruchomienie lokalne:  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

LITERS_PER_BARREL = 158.987
FUELS = ["PB95", "PB98", "ON"]
FUEL_LABEL = {"PB95": "Benzyna 95", "PB98": "Benzyna 98", "ON": "Diesel (ON)"}
DATA_DIR = Path(__file__).parent / "data"

COL = {
    "PB95": "#F4B43E", "PB98": "#E8703A", "ON": "#5FB0A8", "brent": "#9AA0AB",
    "accent": "#F5A623", "up": "#E8703A", "down": "#5FB0A8",
    "grid": "rgba(255,255,255,0.06)", "muted": "#8A8275",
}


# --------------------------------------------------------------------------- #
#  Generator realistycznych danych przykladowych (gdy brak prawdziwych CSV)
# --------------------------------------------------------------------------- #
def generate_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", "2026-06-06", freq="D")
    n = len(dates)

    def interp(anchors: dict) -> np.ndarray:
        ax = pd.to_datetime(list(anchors.keys())).astype("int64").to_numpy()
        ay = np.array(list(anchors.values()), dtype=float)
        return np.interp(dates.astype("int64").to_numpy(), ax, ay)

    def ar1(sd: float, rho: float = 0.92) -> np.ndarray:
        e = rng.normal(0, sd, n)
        out = np.zeros(n)
        for i in range(1, n):
            out[i] = rho * out[i - 1] + e[i]
        return out

    brent = interp({
        "2020-01-01": 64, "2020-04-21": 20, "2020-12-01": 50, "2021-06-01": 73,
        "2021-12-01": 75, "2022-03-08": 124, "2022-06-01": 117, "2022-09-01": 90,
        "2022-12-01": 81, "2023-06-01": 75, "2023-12-01": 77, "2024-06-01": 82,
        "2024-12-01": 73, "2025-06-01": 70, "2025-12-01": 72, "2026-06-06": 71,
    }) + ar1(0.6)
    brent = np.clip(brent, 15, None)

    usd = interp({
        "2020-01-01": 3.80, "2020-04-01": 4.18, "2020-12-01": 3.75, "2021-12-01": 4.05,
        "2022-03-01": 4.30, "2022-10-15": 5.02, "2022-12-01": 4.40, "2023-06-01": 4.10,
        "2023-12-01": 3.97, "2024-06-01": 4.00, "2024-12-01": 4.10, "2025-06-01": 3.80,
        "2026-06-06": 3.78,
    }) + ar1(0.012)

    crude_pln_l = brent * usd / LITERS_PER_BARREL
    tarcza = np.where((dates >= "2022-02-01") & (dates <= "2022-12-31"), -0.70, 0.0)

    LAG = 11
    crude_lagged = np.concatenate([np.full(LAG, crude_pln_l[0]), crude_pln_l[:-LAG]])
    doy = dates.dayofyear.to_numpy()
    season_summer = 0.10 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    season_winter = 0.12 * np.cos(2 * np.pi * doy / 365.25)

    def simulate(base, passthrough, season, up_speed, down_speed, noise_sd):
        target = base + passthrough * crude_lagged + season + tarcza
        price = np.zeros(n)
        price[0] = target[0]
        for i in range(1, n):
            gap = target[i] - price[i - 1]
            price[i] = price[i - 1] + (up_speed if gap > 0 else down_speed) * gap
        return price + ar1(noise_sd, rho=0.6)

    pb95 = simulate(3.70, 1.45, season_summer, 0.16, 0.07, 0.015)
    on = simulate(4.30, 1.58, season_winter, 0.17, 0.08, 0.018)
    pb98 = pb95 + 0.82 + ar1(0.01, rho=0.6)

    df = pd.DataFrame({
        "date": dates, "PB95": pb95.round(3), "PB98": pb98.round(3), "ON": on.round(3),
        "usdpln": usd.round(4), "brent_usd": brent.round(2),
    })
    df["brent_pln_l"] = df["brent_usd"] * df["usdpln"] / LITERS_PER_BARREL
    return df


def load_from_csv() -> pd.DataFrame:
    fuel = pd.read_csv(DATA_DIR / "fuel_prices.csv", parse_dates=["date"])
    usd = pd.read_csv(DATA_DIR / "usdpln.csv", parse_dates=["date"])
    brent = pd.read_csv(DATA_DIR / "brent.csv", parse_dates=["date"])
    df = (fuel.merge(usd, on="date", how="outer").merge(brent, on="date", how="outer")
          .sort_values("date").set_index("date").asfreq("D").ffill().dropna())
    df["brent_pln_l"] = df["brent_usd"] * df["usdpln"] / LITERS_PER_BARREL
    return df.reset_index()


# --------------------------------------------------------------------------- #
#  Pobieranie PRAWDZIWYCH danych z API (dziala na serwerze z dostepem do sieci)
# --------------------------------------------------------------------------- #
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FuelDashboard/1.0; +streamlit)"}


def _fetch_fuel() -> pd.DataFrame:
    """Ceny detaliczne z publicznego API paliwo.today."""
    frames = []
    for fuel in FUELS:
        r = requests.get(f"https://api.paliwo.today/api/prices?type={fuel}",
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        rec = pd.DataFrame(r.json())
        rec["date"] = pd.to_datetime(rec["date"]).dt.tz_localize(None).dt.normalize()
        rec["price"] = pd.to_numeric(rec["price"], errors="coerce")
        frames.append(rec[["date", "price"]].rename(columns={"price": fuel}).set_index("date"))
    return pd.concat(frames, axis=1).sort_index()


def _fetch_usd() -> pd.DataFrame:
    """Kurs sredni USD/PLN z API NBP (tabela A, max 367 dni na zapytanie)."""
    out, start, today = [], date(2020, 1, 1), date.today()
    while start < today:
        end = min(start + timedelta(days=366), today)
        r = requests.get(f"https://api.nbp.pl/api/exchangerates/rates/a/usd/"
                         f"{start}/{end}/?format=json", headers=HEADERS, timeout=30)
        if r.status_code == 200:
            out += [{"date": x["effectiveDate"], "usdpln": x["mid"]} for x in r.json()["rates"]]
        start = end + timedelta(days=1)
    d = pd.DataFrame(out)
    d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates("date").set_index("date")


def _fetch_brent() -> pd.DataFrame:
    """Notowania Brent (USD/bbl): najpierw yfinance, w razie problemu Stooq (CSV)."""
    # 1) yfinance (Yahoo)
    try:
        import yfinance as yf
        data = yf.download("BZ=F", start="2020-01-01", progress=False, auto_adjust=True)
        close = data["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        d = close.reset_index()
        d.columns = ["date", "brent_usd"]
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
        d = d.dropna()
        if len(d) > 100:
            return d.set_index("date")
    except Exception:
        pass
    # 2) Stooq - darmowy CSV bez klucza (cb.f = Brent)
    from io import StringIO
    r = requests.get("https://stooq.com/q/d/l/?s=cb.f&i=d", headers=HEADERS, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(StringIO(r.text))
    d = d.rename(columns={"Date": "date", "Close": "brent_usd"})[["date", "brent_usd"]]
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["brent_usd"] = pd.to_numeric(d["brent_usd"], errors="coerce")
    return d.dropna().set_index("date")


@st.cache_data(ttl=21600, show_spinner="Pobieram dane z internetu...")
def get_data() -> tuple[pd.DataFrame, str, str]:
    """Probuje pobrac PRAWDZIWE dane (z diagnostyka per-zrodlo); w razie problemu - przykladowe."""
    errors, fuel, usd, brent = [], None, None, None
    try:
        fuel = _fetch_fuel()
    except Exception as e:  # noqa: BLE001
        errors.append(f"ceny paliw (paliwo.today): {type(e).__name__}: {e}")
    try:
        usd = _fetch_usd()
    except Exception as e:  # noqa: BLE001
        errors.append(f"kurs USD/PLN (NBP): {type(e).__name__}: {e}")
    try:
        brent = _fetch_brent()
    except Exception as e:  # noqa: BLE001
        errors.append(f"ropa Brent: {type(e).__name__}: {e}")

    if fuel is not None and usd is not None and brent is not None:
        try:
            df = (fuel.join(usd, how="outer").join(brent, how="outer")
                  .sort_index().asfreq("D").ffill().dropna().reset_index()
                  .rename(columns={"index": "date"}))
            df["brent_pln_l"] = df["brent_usd"] * df["usdpln"] / LITERS_PER_BARREL
            if len(df) > 100:
                return df, "live", f"do {df['date'].max():%Y-%m-%d}"
            errors.append(f"po scaleniu zbyt malo danych ({len(df)} dni)")
        except Exception as e:  # noqa: BLE001
            errors.append(f"scalanie danych: {type(e).__name__}: {e}")

    return generate_sample(), "sample", (" | ".join(errors) or "nieznany blad")[:600]


# --------------------------------------------------------------------------- #
#  Analiza
# --------------------------------------------------------------------------- #
def lag_correlation(df: pd.DataFrame, fuel: str, max_lag: int = 30) -> pd.DataFrame:
    d_price = df[fuel].diff()
    d_crude = df["brent_pln_l"].diff()
    return pd.DataFrame([{"lag": l, "corr": d_price.corr(d_crude.shift(l))}
                         for l in range(max_lag + 1)])


def best_lag(df: pd.DataFrame, fuel: str, max_lag: int = 30) -> int:
    cc = lag_correlation(df, fuel, max_lag)
    return int(cc.loc[cc["corr"].idxmax(), "lag"])


def asymmetry(df: pd.DataFrame, fuel: str, lag: int) -> dict:
    d_price = df[fuel].diff()
    d_crude = df["brent_pln_l"].diff().shift(lag)
    data = pd.DataFrame({"y": d_price, "up": d_crude.clip(lower=0),
                         "down": d_crude.clip(upper=0)}).dropna()
    model = LinearRegression().fit(data[["up", "down"]].values, data["y"].values)
    bu, bd = model.coef_
    return {"beta_up": float(bu), "beta_down": float(bd),
            "ratio": float(bu / bd) if bd else float("nan"), "asymmetric": bool(bu > bd)}


def _seasonal_features(dates: pd.Series) -> np.ndarray:
    doy = dates.dt.dayofyear.values
    return np.column_stack([np.sin(2 * np.pi * doy / 365.25),
                            np.cos(2 * np.pi * doy / 365.25)])


def forecast(df: pd.DataFrame, fuel: str, lag: int, horizon: int = 30) -> pd.DataFrame:
    work = df.copy()
    work["crude_lag"] = work["brent_pln_l"].shift(lag)
    work["usd_lag"] = work["usdpln"].shift(lag)
    work = work.dropna(subset=["crude_lag", "usd_lag", fuel])

    X = np.column_stack([work["crude_lag"].values, work["usd_lag"].values,
                         _seasonal_features(work["date"])])
    y = work[fuel].values
    model = LinearRegression().fit(X, y)
    fitted = model.predict(X)
    resid_sd = float(np.std(y - fitted, ddof=1))

    last_crude = float(df["brent_pln_l"].iloc[-1])
    last_usd = float(df["usdpln"].iloc[-1])
    future = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    Xf = np.column_stack([np.full(horizon, last_crude), np.full(horizon, last_usd),
                          _seasonal_features(pd.Series(future))])
    yhat = model.predict(Xf)

    hist = pd.DataFrame({"date": work["date"].values, "value": y, "kind": "historia"})
    fut = pd.DataFrame({"date": future, "value": yhat,
                        "lo": yhat - 1.96 * resid_sd, "hi": yhat + 1.96 * resid_sd,
                        "kind": "prognoza"})
    out = pd.concat([hist, fut], ignore_index=True)
    out.attrs["r2"] = float(model.score(X, y))
    return out


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Ceny paliw w Polsce - analiza", page_icon="\u26fd",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; letter-spacing: -0.02em; }
.big-title { font-size: 2.5rem; font-weight: 700; line-height: 1.05; margin: 0; }
.subtitle { color: #8A8275; font-size: 1.05rem; margin-top: .35rem; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.insight { background: linear-gradient(135deg, rgba(245,166,35,.12), rgba(232,112,58,.06));
           border-left:4px solid #F5A623; border-radius:10px; padding:1rem 1.2rem; margin:.5rem 0 1rem; }
.insight b { color:#F5A623; }
.src { color:#8A8275; font-size:.85rem; }
hr { border-color: rgba(255,255,255,.07); }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#ECE7DF", size=13),
    margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


def style_axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(gridcolor=COL["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COL["grid"], zeroline=False)
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


df_all, source, info = get_data()

with st.sidebar:
    st.markdown("### \u26fd Ustawienia")
    if source == "live":
        st.caption(f"\U0001F7E2 Dane na zywo ({info})")
    else:
        st.warning("Pokazuje dane PRZYKLADOWE - pobranie danych z internetu sie nie powiodlo.")
        with st.expander("Dlaczego? (szczegoly bledu)"):
            st.code(info)
        if st.button("\U0001F504 Sprobuj pobrac ponownie"):
            st.cache_data.clear()
            st.rerun()
    fuels = st.multiselect("Rodzaje paliwa", FUELS, default=FUELS,
                           format_func=lambda f: FUEL_LABEL[f]) or ["PB95"]
    dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
    date_range = st.slider("Zakres dat", dmin, dmax, (dmin, dmax), format="YYYY-MM")
    focus = st.selectbox("Paliwo do analizy szczegolowej", FUELS,
                         format_func=lambda f: FUEL_LABEL[f])
    horizon = st.slider("Horyzont prognozy (dni)", 7, 90, 30, step=7)
    st.markdown("---")
    st.markdown("<span class='src'>Zrodla: ceny - paliwo.today; kurs USD/PLN - NBP; "
                "Brent - Yahoo Finance.</span>", unsafe_allow_html=True)

mask = (df_all["date"].dt.date >= date_range[0]) & (df_all["date"].dt.date <= date_range[1])
df = df_all.loc[mask].reset_index(drop=True)

st.markdown("<p class='big-title'>Ceny paliw w Polsce</p>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>Trendy, sezonowosc i reakcja cen na rynek ropy "
            f"&nbsp;&middot;&nbsp; {dmin:%Y} - {dmax:%Y}</p>", unsafe_allow_html=True)

latest = df_all.iloc[-1]
week_ago = df_all.iloc[-8] if len(df_all) > 8 else df_all.iloc[0]
cols = st.columns(5)
for c, f in zip(cols[:3], FUELS):
    c.metric(FUEL_LABEL[f], f"{latest[f]:.2f} zl/l",
             f"{latest[f] - week_ago[f]:+.2f} zl/tydz.", delta_color="inverse")
cols[3].metric("Ropa Brent", f"{latest['brent_usd']:.1f} $/bbl",
               f"{latest['brent_usd'] - week_ago['brent_usd']:+.1f}", delta_color="off")
cols[4].metric("Kurs USD/PLN", f"{latest['usdpln']:.3f}",
               f"{latest['usdpln'] - week_ago['usdpln']:+.3f}", delta_color="off")
st.markdown("---")

t1, t2, t3, t4, t5, t6 = st.tabs(
    ["Trend", "Ropa vs cena", "Sezonowosc", "Opoznienie", "Asymetria", "Prognoza"])

with t1:
    st.subheader("Ceny detaliczne w czasie")
    fig = go.Figure()
    for f in fuels:
        fig.add_trace(go.Scatter(x=df["date"], y=df[f], name=FUEL_LABEL[f],
                                 line=dict(color=COL[f], width=2)))
    fig.update_yaxes(title="zl / litr")
    st.plotly_chart(style_axes(fig), use_container_width=True)
    pb = df_all["PB95"]
    st.markdown(f"<div class='insight'>Od poczatku 2020 r. benzyna 95 wahala sie od "
                f"<b>{pb.min():.2f}</b> do <b>{pb.max():.2f} zl/l</b>. Szczyt przypadl na "
                f"lato 2022 (wojna w Ukrainie, slaby zloty), a obnizka VAT przejsciowo go "
                f"stlumila.</div>", unsafe_allow_html=True)

with t2:
    st.subheader("Ile w cenie to surowiec, a ile podatki i marza")
    st.caption("Brent przeliczony na PLN/l (1 barylka = 159 l) na tle ceny detalicznej. "
               "Roznica to akcyza, oplata paliwowa, VAT, marza i przerob.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df[focus], name=f"{FUEL_LABEL[focus]} (detal)",
                             line=dict(color=COL[focus], width=2.2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["brent_pln_l"], name="Surowiec (Brent w PLN/l)",
                             line=dict(color=COL["brent"], width=1.6), fill="tozeroy",
                             fillcolor="rgba(154,160,171,.12)"))
    fig.update_yaxes(title="zl / litr")
    st.plotly_chart(style_axes(fig), use_container_width=True)
    a, b = st.columns(2)
    a.metric("Sredni narzut ponad surowiec", f"{(df_all[focus] - df_all['brent_pln_l']).mean():.2f} zl/l")
    b.metric("Udzial surowca w cenie", f"{(df_all['brent_pln_l'] / df_all[focus]).mean() * 100:.0f}%")

with t3:
    st.subheader(f"Sezonowosc - {FUEL_LABEL[focus]}")
    tmp = df_all.copy()
    tmp["month"] = tmp["date"].dt.month
    tmp["dev"] = tmp.groupby(tmp["date"].dt.year)[focus].transform(lambda s: s - s.mean())
    monthly = tmp.groupby("month")["dev"].mean()
    months_pl = ["Sty", "Lut", "Mar", "Kwi", "Maj", "Cze", "Lip", "Sie", "Wrz", "Paz", "Lis", "Gru"]
    fig = go.Figure(go.Bar(x=months_pl, y=monthly.values,
                           marker_color=[COL["accent"] if v >= 0 else COL["ON"] for v in monthly.values]))
    fig.update_yaxes(title="odchylenie od sredniej rocznej (zl/l)")
    st.plotly_chart(style_axes(fig), use_container_width=True)
    st.markdown(f"<div class='insight'>Po usunieciu trendu widac powtarzalny wzorzec: "
                f"najdrozej zwykle w <b>{months_pl[int(monthly.idxmax()) - 1]}</b>, "
                f"najtaniej w <b>{months_pl[int(monthly.idxmin()) - 1]}</b>.</div>",
                unsafe_allow_html=True)
    with st.expander("Dekompozycja szeregu (trend / sezon / reszta)"):
        s = df_all.set_index("date")[focus].asfreq("D").interpolate()
        dec = seasonal_decompose(s, period=365, model="additive", extrapolate_trend="freq")
        for comp, name, color in [(dec.trend, "Trend", COL[focus]),
                                  (dec.seasonal, "Skladnik sezonowy", COL["accent"]),
                                  (dec.resid, "Reszta", COL["muted"])]:
            f2 = go.Figure(go.Scatter(x=comp.index, y=comp.values, line=dict(color=color, width=1.4)))
            f2.update_layout(title=name, height=220)
            st.plotly_chart(style_axes(f2), use_container_width=True)

with t4:
    st.subheader(f"Z jakim opoznieniem stacje reaguja na rynek - {FUEL_LABEL[focus]}")
    st.caption("Korelacja dziennej zmiany ceny paliwa ze zmiana ceny surowca sprzed N dni.")
    cc = lag_correlation(df_all, focus, 30)
    lag = int(cc.loc[cc["corr"].idxmax(), "lag"])
    fig = go.Figure(go.Bar(x=cc["lag"], y=cc["corr"],
                           marker_color=[COL["accent"] if l == lag else "rgba(245,166,35,.35)"
                                         for l in cc["lag"]]))
    fig.update_xaxes(title="opoznienie (dni)")
    fig.update_yaxes(title="korelacja zmian")
    st.plotly_chart(style_axes(fig), use_container_width=True)
    st.markdown(f"<div class='insight'>Ceny na stacjach najsilniej odzwierciedlaja rynek ropy "
                f"z opoznieniem okolo <b>{lag} dni</b>. To opoznienie wykorzystuje model "
                f"prognostyczny.</div>", unsafe_allow_html=True)

with t5:
    st.subheader("Efekt rakiety i piorka")
    st.caption("Czy stacje szybciej podnosza ceny przy drozejacej ropie, niz obnizaja "
               "przy taniejacej?")
    lag = best_lag(df_all, focus)
    asym = asymmetry(df_all, focus, lag)
    fig = go.Figure(go.Bar(x=["Reakcja na WZROST ropy", "Reakcja na SPADEK ropy"],
                           y=[asym["beta_up"], asym["beta_down"]],
                           marker_color=[COL["up"], COL["down"]],
                           text=[f"{asym['beta_up']:.3f}", f"{asym['beta_down']:.3f}"],
                           textposition="outside"))
    fig.update_yaxes(title="sila przelozenia na cene")
    st.plotly_chart(style_axes(fig), use_container_width=True)
    if asym["asymmetric"]:
        st.markdown(f"<div class='insight'>Potwierdzony efekt <b>rakiety i piorka</b>: "
                    f"przelozenie wzrostow ropy jest ok. <b>{asym['ratio']:.1f}x</b> silniejsze "
                    f"niz spadkow. Drozeje szybko, tanieje powoli.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight'>W tym zakresie reakcja na wzrosty i spadki ropy "
                    "jest zblizona.</div>", unsafe_allow_html=True)

with t6:
    st.subheader(f"Prognoza ceny - {FUEL_LABEL[focus]} (+{horizon} dni)")
    st.caption("Model: cena = f(surowiec sprzed N dni, kurs USD/PLN, sezonowosc). "
               "Prognoza to scenariusz przy stalych Brent i kursie.")
    lag = best_lag(df_all, focus)
    fc = forecast(df_all, focus, lag, horizon)
    hist = fc[fc.kind == "historia"]
    fut = fc[fc.kind == "prognoza"]
    show_hist = hist[hist["date"] >= hist["date"].max() - pd.Timedelta(days=365)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(fut["date"]) + list(fut["date"][::-1]),
                             y=list(fut["hi"]) + list(fut["lo"][::-1]), fill="toself",
                             fillcolor="rgba(245,166,35,.15)", line=dict(width=0),
                             name="przedzial 95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=show_hist["date"], y=show_hist["value"], name="historia",
                             line=dict(color=COL[focus], width=2)))
    fig.add_trace(go.Scatter(x=fut["date"], y=fut["value"], name="prognoza",
                             line=dict(color=COL["accent"], width=2.4, dash="dot")))
    fig.update_yaxes(title="zl / litr")
    st.plotly_chart(style_axes(fig), use_container_width=True)
    a, b = st.columns(2)
    a.metric(f"Prognoza za {horizon} dni", f"{fut['value'].iloc[-1]:.2f} zl/l",
             f"{fut['value'].iloc[-1] - latest[focus]:+.2f} zl", delta_color="inverse")
    b.metric("Dopasowanie modelu (R2)", f"{fc.attrs['r2']:.3f}")
    st.caption("Prognoza pogladowa - nie uwzglednia szokow geopolitycznych ani zmian "
               "podatkowych. Sluzy demonstracji metody.")

st.markdown("---")
st.markdown("<span class='src'>Projekt zaliczeniowy &middot; analiza danych + komponent "
            "predykcyjny. Metody: cross-correlation, regresja asymetryczna, dekompozycja "
            "sezonowa, regresja liniowa z opoznieniem.</span>", unsafe_allow_html=True)
