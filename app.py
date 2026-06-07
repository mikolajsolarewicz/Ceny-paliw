"""
Analiza cen paliw w Polsce - dashboard Streamlit (wersja tygodniowa).

Dane:
  - HISTORIA (od 2020) z pliku data/fuel_weekly.csv - tygodniowe ceny detaliczne
    benzyny 95 i oleju napedowego w PLN, zrodlo: Biuletyn Naftowy Komisji Europejskiej.
  - NA ZYWO: ropa Brent (Yahoo/Stooq), kurs USD/PLN (NBP) oraz najswiezsze ceny paliw
    (paliwo.today) doklejane po dacie konca pliku historii.
Jesli pobranie Brent/kursu zawiedzie, aplikacja pokazuje awaryjnie dane przykladowe.
"""

from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

LITERS_PER_BARREL = 158.987
FUELS = ["PB95", "ON"]
FUEL_LABEL = {"PB95": "Benzyna 95", "ON": "Diesel (ON)"}
DATA_DIR = Path(__file__).parent / "data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FuelDashboard/1.0)"}

COL = {"PB95": "#F4B43E", "ON": "#5FB0A8", "brent": "#9AA0AB", "accent": "#F5A623",
       "up": "#E8703A", "down": "#5FB0A8", "grid": "rgba(255,255,255,0.06)", "muted": "#8A8275"}


# --------------------------------------------------------------------------- #
#  Pobieranie danych
# --------------------------------------------------------------------------- #
def load_fuel_weekly() -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).parent / "fuel_weekly.csv", parse_dates=["date"])
    return df.sort_values("date")[["date", "PB95", "ON"]]


def _fetch_brent() -> pd.DataFrame:
    try:
        import yfinance as yf
        data = yf.download("BZ=F", start="2019-12-01", progress=False, auto_adjust=True)
        close = data["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        d = close.reset_index()
        d.columns = ["date", "brent_usd"]
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
        d = d.dropna()
        if len(d) > 100:
            return d.sort_values("date")
    except Exception:
        pass
    r = requests.get("https://stooq.com/q/d/l/?s=cb.f&i=d", headers=HEADERS, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(StringIO(r.text)).rename(columns={"Date": "date", "Close": "brent_usd"})
    d = d[["date", "brent_usd"]]
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["brent_usd"] = pd.to_numeric(d["brent_usd"], errors="coerce")
    return d.dropna().sort_values("date")


def _fetch_usd() -> pd.DataFrame:
    out, start, today = [], date(2019, 12, 1), date.today()
    while start < today:
        end = min(start + timedelta(days=366), today)
        r = requests.get(f"https://api.nbp.pl/api/exchangerates/rates/a/usd/"
                         f"{start}/{end}/?format=json", headers=HEADERS, timeout=30)
        if r.status_code == 200:
            out += [{"date": x["effectiveDate"], "usdpln": x["mid"]} for x in r.json()["rates"]]
        start = end + timedelta(days=1)
    d = pd.DataFrame(out)
    d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates("date").sort_values("date")


def _fetch_paliwo_recent() -> pd.DataFrame:
    frames = []
    for fuel in FUELS:
        r = requests.get(f"https://api.paliwo.today/api/prices?type={fuel}",
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        rec = pd.DataFrame(r.json())
        rec["date"] = pd.to_datetime(rec["date"]).dt.tz_localize(None).dt.normalize()
        rec["price"] = pd.to_numeric(rec["price"], errors="coerce")
        frames.append(rec[["date", "price"]].rename(columns={"price": fuel}).set_index("date"))
    return pd.concat(frames, axis=1).sort_index().reset_index()


def generate_sample() -> pd.DataFrame:
    """Awaryjny zbior dzienny (syntetyczny), gdy pobranie Brent/kursu sie nie powiedzie."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", "2026-06-06", freq="D")
    n = len(dates)

    def interp(a):
        ax = pd.to_datetime(list(a)).astype("int64").to_numpy()
        return np.interp(dates.astype("int64").to_numpy(), ax, np.array(list(a.values()), float))

    def ar1(sd, rho=0.92):
        e = rng.normal(0, sd, n); o = np.zeros(n)
        for i in range(1, n):
            o[i] = rho * o[i - 1] + e[i]
        return o

    brent = np.clip(interp({"2020-01-01": 64, "2020-04-21": 20, "2022-03-08": 124,
                            "2022-09-01": 90, "2024-06-01": 82, "2026-06-06": 92}) + ar1(0.6), 15, None)
    usd = interp({"2020-01-01": 3.80, "2022-10-15": 5.02, "2024-06-01": 4.00, "2026-06-06": 3.64}) + ar1(0.012)
    crude = brent * usd / LITERS_PER_BARREL
    lagc = np.concatenate([np.full(11, crude[0]), crude[:-11]])

    def sim(base, pt, up, dn):
        tgt = base + pt * lagc; p = np.zeros(n); p[0] = tgt[0]
        for i in range(1, n):
            g = tgt[i] - p[i - 1]; p[i] = p[i - 1] + (up if g > 0 else dn) * g
        return p + ar1(0.015, 0.6)

    df = pd.DataFrame({"date": dates, "PB95": sim(3.70, 1.45, .16, .07).round(3),
                       "ON": sim(4.30, 1.58, .17, .08).round(3),
                       "usdpln": usd.round(4), "brent_usd": brent.round(2)})
    df["brent_pln_l"] = df["brent_usd"] * df["usdpln"] / LITERS_PER_BARREL
    return df


@st.cache_data(ttl=21600, show_spinner="Wczytuje dane...")
def get_data() -> tuple[pd.DataFrame, str, str]:
    errors = []
    try:
        fuel = load_fuel_weekly()
    except Exception as e:  # noqa: BLE001
        return generate_sample(), "sample", f"brak pliku historii: {e}"
    if len(fuel) < 50:
        return generate_sample(), "sample", "plik historii zbyt krotki"

    # doklejenie najswiezszych tygodni z paliwo.today (po koncu historii)
    try:
        rec = _fetch_paliwo_recent()
        rec = rec[rec["date"] > fuel["date"].max()]
        if len(rec):
            rw = (rec.set_index("date").resample("W-MON").mean().dropna().reset_index())
            fuel = pd.concat([fuel, rw[["date", "PB95", "ON"]]], ignore_index=True)
    except Exception:  # noqa: BLE001
        pass
    fuel = fuel.dropna().sort_values("date")

    try:
        brent = _fetch_brent()
        usd = _fetch_usd()
    except Exception as e:  # noqa: BLE001
        return generate_sample(), "sample", f"Brent/kurs: {type(e).__name__}: {e}"

    try:
        for _d in (fuel, brent, usd):
            _d["date"] = _d["date"].astype("datetime64[ns]")
        df = pd.merge_asof(fuel, brent, on="date")
        df = pd.merge_asof(df, usd, on="date")
        df["brent_pln_l"] = df["brent_usd"] * df["usdpln"] / LITERS_PER_BARREL
        df = df.dropna().reset_index(drop=True)
        if len(df) >= 50:
            return df, "live", f"{df['date'].min():%m.%Y}\u2013{df['date'].max():%m.%Y}, {len(df)} tyg."
        errors.append(f"po scaleniu zbyt malo ({len(df)})")
    except Exception as e:  # noqa: BLE001
        errors.append(f"scalanie: {type(e).__name__}: {e}")
    return generate_sample(), "sample", (" | ".join(errors) or "nieznany blad")[:500]


# --------------------------------------------------------------------------- #
#  Analiza (dziala na danych dziennych i tygodniowych - operuje na krokach)
# --------------------------------------------------------------------------- #
def lag_correlation(df, fuel, max_lag):
    dp, dc = df[fuel].diff(), df["brent_pln_l"].diff()
    return pd.DataFrame([{"lag": l, "corr": dp.corr(dc.shift(l))} for l in range(max_lag + 1)])


def best_lag(df, fuel, max_lag):
    cc = lag_correlation(df, fuel, max_lag)
    return int(cc.loc[cc["corr"].idxmax(), "lag"])


def asymmetry(df, fuel, lag):
    dp = df[fuel].diff()
    dc = df["brent_pln_l"].diff().shift(lag)
    data = pd.DataFrame({"y": dp, "up": dc.clip(lower=0), "down": dc.clip(upper=0)}).dropna()
    m = LinearRegression().fit(data[["up", "down"]].values, data["y"].values)
    bu, bd = m.coef_
    return {"beta_up": float(bu), "beta_down": float(bd),
            "ratio": float(bu / bd) if bd else float("nan"), "asymmetric": bool(bu > bd)}


def _seas(dates):
    doy = dates.dt.dayofyear.values
    return np.column_stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])


def forecast(df, fuel, lag, horizon, freq):
    w = df.copy()
    w["cl"] = w["brent_pln_l"].shift(lag)
    w["ul"] = w["usdpln"].shift(lag)
    w = w.dropna(subset=["cl", "ul", fuel])
    X = np.column_stack([w["cl"].values, w["ul"].values, _seas(w["date"])])
    y = w[fuel].values
    m = LinearRegression().fit(X, y)
    sd = float(np.std(y - m.predict(X), ddof=1))
    step = pd.Timedelta(days=7) if freq.startswith("W") else pd.Timedelta(days=1)
    fut = pd.date_range(df["date"].iloc[-1] + step, periods=horizon, freq=freq)
    Xf = np.column_stack([np.full(horizon, df["brent_pln_l"].iloc[-1]),
                          np.full(horizon, df["usdpln"].iloc[-1]), _seas(pd.Series(fut))])
    yh = m.predict(Xf)
    hist = pd.DataFrame({"date": w["date"].values, "value": y, "kind": "historia"})
    f = pd.DataFrame({"date": fut, "value": yh, "lo": yh - 1.96 * sd, "hi": yh + 1.96 * sd,
                      "kind": "prognoza"})
    out = pd.concat([hist, f], ignore_index=True)
    out.attrs["r2"] = float(m.score(X, y))
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
h1,h2,h3 { font-family:'IBM Plex Sans',sans-serif; letter-spacing:-0.02em; }
.big-title { font-size:2.5rem; font-weight:700; line-height:1.05; margin:0; }
.subtitle { color:#8A8275; font-size:1.05rem; margin-top:.35rem; }
[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace; font-weight:600; }
.insight { background:linear-gradient(135deg,rgba(245,166,35,.12),rgba(232,112,58,.06));
           border-left:4px solid #F5A623; border-radius:10px; padding:1rem 1.2rem; margin:.5rem 0 1rem; }
.insight b { color:#F5A623; } .src { color:#8A8275; font-size:.85rem; }
hr { border-color:rgba(255,255,255,.07); }
</style>""", unsafe_allow_html=True)

PLOTLY = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font=dict(family="IBM Plex Sans, sans-serif", color="#ECE7DF", size=13),
              margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


def sx(fig):
    fig.update_xaxes(gridcolor=COL["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COL["grid"], zeroline=False)
    fig.update_layout(**PLOTLY)
    return fig


df_all, source, info = get_data()

steps = df_all["date"].diff().dt.days.dropna()
WEEKLY = bool(steps.median() >= 4) if len(steps) else False
PERIOD = 52 if WEEKLY else 365
MAXLAG = 12 if WEEKLY else 28
UNIT = "tyg." if WEEKLY else "dni"
FREQ = "W-MON" if WEEKLY else "D"

with st.sidebar:
    st.markdown("### \u26fd Ustawienia")
    if source == "live":
        st.caption(f"\U0001F7E2 Dane prawdziwe ({info})")
    else:
        st.warning("Dane PRZYKLADOWE - pobranie danych sie nie powiodlo.")
        with st.expander("Dlaczego? (szczegoly)"):
            st.code(info)
        if st.button("\U0001F504 Sprobuj ponownie"):
            st.cache_data.clear(); st.rerun()
    fuels = st.multiselect("Rodzaje paliwa", FUELS, default=FUELS,
                           format_func=lambda f: FUEL_LABEL[f]) or ["PB95"]
    dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
    date_range = st.slider("Zakres dat", dmin, dmax, (dmin, dmax), format="YYYY-MM")
    focus = st.selectbox("Paliwo do analizy szczegolowej", FUELS, format_func=lambda f: FUEL_LABEL[f])
    if WEEKLY:
        horizon = st.slider("Horyzont prognozy (tygodnie)", 4, 26, 12, step=2)
    else:
        horizon = st.slider("Horyzont prognozy (dni)", 7, 90, 30, step=7)
    st.markdown("---")
    st.markdown("<span class='src'>Zrodla: ceny paliw - Biuletyn Naftowy UE (tygodniowo) + "
                "paliwo.today; Brent - Yahoo/Stooq; kurs USD/PLN - NBP.</span>", unsafe_allow_html=True)

mask = (df_all["date"].dt.date >= date_range[0]) & (df_all["date"].dt.date <= date_range[1])
df = df_all.loc[mask].reset_index(drop=True)

st.markdown("<p class='big-title'>Ceny paliw w Polsce</p>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>Trendy, sezonowosc i reakcja cen na rynek ropy "
            f"&nbsp;&middot;&nbsp; {dmin:%d.%m.%Y} \u2013 {dmax:%d.%m.%Y}</p>", unsafe_allow_html=True)

latest = df_all.iloc[-1]
ref = df_all[df_all["date"] <= latest["date"] - pd.Timedelta(days=7)]
prev = ref.iloc[-1] if len(ref) else df_all.iloc[0]
cols = st.columns(4)
for c, f in zip(cols[:2], FUELS):
    c.metric(FUEL_LABEL[f], f"{latest[f]:.2f} zl/l",
             f"{latest[f] - prev[f]:+.2f} zl/tydz.", delta_color="inverse")
cols[2].metric("Ropa Brent", f"{latest['brent_usd']:.1f} $/bbl",
               f"{latest['brent_usd'] - prev['brent_usd']:+.1f}", delta_color="off")
cols[3].metric("Kurs USD/PLN", f"{latest['usdpln']:.3f}",
               f"{latest['usdpln'] - prev['usdpln']:+.3f}", delta_color="off")
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
    st.plotly_chart(sx(fig), use_container_width=True)
    pb = df_all["PB95"]
    lo_d, hi_d = df_all.loc[pb.idxmin(), "date"], df_all.loc[pb.idxmax(), "date"]
    st.markdown(f"<div class='insight'>W analizowanym okresie benzyna 95 wahala sie od "
                f"<b>{pb.min():.2f} zl/l</b> ({lo_d:%m.%Y}) do <b>{pb.max():.2f} zl/l</b> "
                f"({hi_d:%m.%Y}). Szczyt to skutek wojny w Ukrainie i slabego zlotego.</div>",
                unsafe_allow_html=True)

with t2:
    st.subheader("Ile w cenie to surowiec, a ile podatki i marza")
    st.caption("Brent przeliczony na PLN/l (1 barylka = 159 l) na tle ceny detalicznej.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df[focus], name=f"{FUEL_LABEL[focus]} (detal)",
                             line=dict(color=COL[focus], width=2.2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["brent_pln_l"], name="Surowiec (Brent w PLN/l)",
                             line=dict(color=COL["brent"], width=1.6), fill="tozeroy",
                             fillcolor="rgba(154,160,171,.12)"))
    fig.update_yaxes(title="zl / litr")
    st.plotly_chart(sx(fig), use_container_width=True)
    a, b = st.columns(2)
    a.metric("Sredni narzut ponad surowiec", f"{(df_all[focus] - df_all['brent_pln_l']).mean():.2f} zl/l")
    b.metric("Udzial surowca w cenie", f"{(df_all['brent_pln_l'] / df_all[focus]).mean() * 100:.0f}%")

with t3:
    st.subheader(f"Sezonowosc - {FUEL_LABEL[focus]}")
    if len(df_all) < 2 * PERIOD:
        st.info(f"Analiza sezonowosci wymaga co najmniej dwoch pelnych lat danych. "
                f"Dostepny zakres jest na to za krotki.")
    else:
        tmp = df_all.copy()
        tmp["month"] = tmp["date"].dt.month
        tmp["dev"] = tmp.groupby(tmp["date"].dt.year)[focus].transform(lambda s: s - s.mean())
        monthly = tmp.groupby("month")["dev"].mean()
        mn = ["Sty", "Lut", "Mar", "Kwi", "Maj", "Cze", "Lip", "Sie", "Wrz", "Paz", "Lis", "Gru"]
        fig = go.Figure(go.Bar(x=mn, y=monthly.values,
                               marker_color=[COL["accent"] if v >= 0 else COL["ON"] for v in monthly.values]))
        fig.update_yaxes(title="odchylenie od sredniej rocznej (zl/l)")
        st.plotly_chart(sx(fig), use_container_width=True)
        st.markdown(f"<div class='insight'>Po usunieciu trendu: najdrozej zwykle w "
                    f"<b>{mn[int(monthly.idxmax()) - 1]}</b>, najtaniej w "
                    f"<b>{mn[int(monthly.idxmin()) - 1]}</b>.</div>", unsafe_allow_html=True)
        with st.expander("Dekompozycja szeregu (trend / sezon / reszta)"):
            dec = seasonal_decompose(df_all[focus].to_numpy(), period=PERIOD,
                                     model="additive", extrapolate_trend="freq")
            xd = df_all["date"]
            for comp, nm, c in [(dec.trend, "Trend", COL[focus]),
                                (dec.seasonal, "Skladnik sezonowy", COL["accent"]),
                                (dec.resid, "Reszta", COL["muted"])]:
                f2 = go.Figure(go.Scatter(x=xd, y=comp, line=dict(color=c, width=1.4)))
                f2.update_layout(title=nm, height=220)
                st.plotly_chart(sx(f2), use_container_width=True)

with t4:
    st.subheader(f"Z jakim opoznieniem stacje reaguja na rynek - {FUEL_LABEL[focus]}")
    st.caption(f"Korelacja zmiany ceny paliwa ze zmiana ceny surowca sprzed N {UNIT}.")
    cc = lag_correlation(df_all, focus, MAXLAG)
    lag = int(cc.loc[cc["corr"].idxmax(), "lag"])
    fig = go.Figure(go.Bar(x=cc["lag"], y=cc["corr"],
                           marker_color=[COL["accent"] if l == lag else "rgba(245,166,35,.35)" for l in cc["lag"]]))
    fig.update_xaxes(title=f"opoznienie ({UNIT})")
    fig.update_yaxes(title="korelacja zmian")
    st.plotly_chart(sx(fig), use_container_width=True)
    st.markdown(f"<div class='insight'>Ceny najsilniej odzwierciedlaja rynek ropy z opoznieniem "
                f"okolo <b>{lag} {UNIT}</b>.</div>", unsafe_allow_html=True)

with t5:
    st.subheader("Efekt rakiety i piorka")
    st.caption("Czy stacje szybciej podnosza ceny przy drozejacej ropie, niz obnizaja przy taniejacej?")
    lag = best_lag(df_all, focus, MAXLAG)
    asym = asymmetry(df_all, focus, lag)
    fig = go.Figure(go.Bar(x=["Reakcja na WZROST ropy", "Reakcja na SPADEK ropy"],
                           y=[asym["beta_up"], asym["beta_down"]], marker_color=[COL["up"], COL["down"]],
                           text=[f"{asym['beta_up']:.3f}", f"{asym['beta_down']:.3f}"], textposition="outside"))
    fig.update_yaxes(title="sila przelozenia na cene")
    st.plotly_chart(sx(fig), use_container_width=True)
    if asym["asymmetric"]:
        st.markdown(f"<div class='insight'>Potwierdzony efekt <b>rakiety i piorka</b>: przelozenie "
                    f"wzrostow ropy jest ok. <b>{asym['ratio']:.1f}x</b> silniejsze niz spadkow. "
                    f"Drozeje szybko, tanieje powoli.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight'>W tym zakresie reakcja na wzrosty i spadki jest zblizona.</div>",
                    unsafe_allow_html=True)

with t6:
    st.subheader(f"Prognoza ceny - {FUEL_LABEL[focus]} (+{horizon} {UNIT})")
    st.caption("Model: cena = f(surowiec z opoznieniem, kurs USD/PLN, sezonowosc). "
               "Scenariusz przy stalych Brent i kursie.")
    lag = best_lag(df_all, focus, MAXLAG)
    fc = forecast(df_all, focus, lag, horizon, FREQ)
    hist = fc[fc.kind == "historia"]
    fut = fc[fc.kind == "prognoza"]
    show = hist[hist["date"] >= hist["date"].max() - pd.Timedelta(days=540)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(fut["date"]) + list(fut["date"][::-1]),
                             y=list(fut["hi"]) + list(fut["lo"][::-1]), fill="toself",
                             fillcolor="rgba(245,166,35,.15)", line=dict(width=0),
                             name="przedzial 95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=show["date"], y=show["value"], name="historia",
                             line=dict(color=COL[focus], width=2)))
    fig.add_trace(go.Scatter(x=fut["date"], y=fut["value"], name="prognoza",
                             line=dict(color=COL["accent"], width=2.4, dash="dot")))
    fig.update_yaxes(title="zl / litr")
    st.plotly_chart(sx(fig), use_container_width=True)
    a, b = st.columns(2)
    a.metric(f"Prognoza za {horizon} {UNIT}", f"{fut['value'].iloc[-1]:.2f} zl/l",
             f"{fut['value'].iloc[-1] - latest[focus]:+.2f} zl", delta_color="inverse")
    b.metric("Dopasowanie modelu (R2)", f"{fc.attrs['r2']:.3f}")
    st.caption("Prognoza pogladowa - nie uwzglednia szokow geopolitycznych ani zmian podatkowych.")

st.markdown("---")
st.markdown("<span class='src'>Projekt zaliczeniowy &middot; analiza danych + komponent predykcyjny. "
            "Dane historyczne: Biuletyn Naftowy Komisji Europejskiej (ceny tygodniowe od 2020).</span>",
            unsafe_allow_html=True)
