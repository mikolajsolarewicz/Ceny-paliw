"""
app.py — Analiza cen paliw w Polsce (2020–2026)
================================================
Interaktywny dashboard: trend, relacja z ropą Brent i kursem USD/PLN,
sezonowość, OPÓŹNIENIE reakcji cen na stacjach, ASYMETRIA ("rakieta i piórko")
oraz prosta prognoza.

Uruchomienie lokalne:   streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose

import analysis as A

# --------------------------------------------------------------------------- #
#  Konfiguracja strony + motyw
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Ceny paliw w Polsce — analiza",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

COL = {
    "PB95": "#F4B43E",   # amber
    "PB98": "#E8703A",   # burnt orange
    "ON":   "#5FB0A8",   # teal
    "brent": "#9AA0AB",  # grey
    "accent": "#F5A623",
    "up": "#E8703A",
    "down": "#5FB0A8",
    "grid": "rgba(255,255,255,0.06)",
    "muted": "#8A8275",
}
FUEL_LABEL = {"PB95": "Benzyna 95", "PB98": "Benzyna 98", "ON": "Diesel (ON)"}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
    html, body, [class*="css"], .stMarkdown, .stMetric { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; letter-spacing: -0.02em; }
    .big-title { font-size: 2.5rem; font-weight: 700; line-height: 1.05; margin: 0; }
    .subtitle { color: #8A8275; font-size: 1.05rem; margin-top: .35rem; }
    [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
    .kpi-card { background:#1E1B17; border:1px solid rgba(245,166,35,.18);
                border-radius:14px; padding:1rem 1.2rem; }
    .insight { background: linear-gradient(135deg, rgba(245,166,35,.12), rgba(232,112,58,.06));
               border-left:4px solid #F5A623; border-radius:10px; padding:1rem 1.2rem; margin:.5rem 0 1rem; }
    .insight b { color:#F5A623; }
    .src { color:#8A8275; font-size:.85rem; }
    hr { border-color: rgba(255,255,255,.07); }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#ECE7DF", size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


def style_axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(gridcolor=COL["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COL["grid"], zeroline=False)
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# --------------------------------------------------------------------------- #
#  Wczytanie danych (z cache)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Wczytuję dane…")
def get_data() -> pd.DataFrame:
    return A.load_data()


try:
    df_all = get_data()
except FileNotFoundError:
    st.error("Brak danych w katalogu `data/`. Uruchom `python generate_sample.py` "
             "(dane przykładowe) lub `python fetch_data.py` (dane prawdziwe).")
    st.stop()

source = A.data_source()

# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⛽ Ustawienia")
    if source == "live":
        st.success("Dane prawdziwe (paliwo.today · NBP · Brent)")
    elif source == "sample":
        st.warning("Dane PRZYKŁADOWE (syntetyczne). Uruchom `python fetch_data.py`, "
                   "aby pobrać prawdziwe.")
    else:
        st.info(f"Źródło danych: {source}")

    fuels = st.multiselect(
        "Rodzaje paliwa",
        options=A.FUELS,
        default=A.FUELS,
        format_func=lambda f: FUEL_LABEL[f],
    )
    if not fuels:
        fuels = ["PB95"]

    dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
    date_range = st.slider("Zakres dat", min_value=dmin, max_value=dmax,
                           value=(dmin, dmax), format="YYYY-MM")

    focus = st.selectbox("Paliwo do analizy szczegółowej",
                         options=A.FUELS, format_func=lambda f: FUEL_LABEL[f])
    horizon = st.slider("Horyzont prognozy (dni)", 7, 90, 30, step=7)
    st.markdown("---")
    st.markdown(
        "<span class='src'>Źródła: ceny detaliczne — paliwo.today · kurs USD/PLN — "
        "NBP (tabela A) · Brent — Yahoo Finance (BZ=F).</span>",
        unsafe_allow_html=True,
    )

mask = (df_all["date"].dt.date >= date_range[0]) & (df_all["date"].dt.date <= date_range[1])
df = df_all.loc[mask].reset_index(drop=True)

# --------------------------------------------------------------------------- #
#  Nagłówek
# --------------------------------------------------------------------------- #
st.markdown("<p class='big-title'>Ceny paliw w Polsce</p>", unsafe_allow_html=True)
st.markdown(
    f"<p class='subtitle'>Analiza trendów, sezonowości i reakcji cen na rynek ropy "
    f"&nbsp;·&nbsp; {dmin:%Y} – {dmax:%Y}</p>",
    unsafe_allow_html=True,
)

# KPI – ostatnie ceny i zmiana tygodniowa
latest = df_all.iloc[-1]
week_ago = df_all.iloc[-8] if len(df_all) > 8 else df_all.iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
for col, fuel in zip((c1, c2, c3), A.FUELS):
    delta = latest[fuel] - week_ago[fuel]
    col.metric(FUEL_LABEL[fuel], f"{latest[fuel]:.2f} zł/l", f"{delta:+.2f} zł/tydz.",
               delta_color="inverse")
c4.metric("Ropa Brent", f"{latest['brent_usd']:.1f} $/bbl",
          f"{latest['brent_usd'] - week_ago['brent_usd']:+.1f}", delta_color="off")
c5.metric("Kurs USD/PLN", f"{latest['usdpln']:.3f}",
          f"{latest['usdpln'] - week_ago['usdpln']:+.3f}", delta_color="off")

st.markdown("---")

tab_trend, tab_crude, tab_season, tab_lag, tab_asym, tab_fc = st.tabs(
    ["📈 Trend", "🛢️ Ropa vs cena", "🗓️ Sezonowość",
     "⏱️ Opóźnienie", "🚀 Asymetria", "🔮 Prognoza"]
)

# --------------------------------------------------------------------------- #
#  1. Trend
# --------------------------------------------------------------------------- #
with tab_trend:
    st.subheader("Ceny detaliczne w czasie")
    fig = go.Figure()
    for f in fuels:
        fig.add_trace(go.Scatter(x=df["date"], y=df[f], name=FUEL_LABEL[f],
                                 line=dict(color=COL[f], width=2)))
    fig.update_yaxes(title="zł / litr")
    st.plotly_chart(style_axes(fig), use_container_width=True)

    pb = df_all["PB95"]
    st.markdown(
        f"<div class='insight'>Od początku 2020 r. benzyna 95 wahała się od "
        f"<b>{pb.min():.2f}</b> do <b>{pb.max():.2f} zł/l</b>. Najwyższe ceny "
        f"przypadły na lato 2022 (skutek wojny w Ukrainie i słabego złotego), "
        f"a interwencja podatkowa (obniżka VAT) przejściowo je stłumiła.</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
#  2. Ropa vs cena — gdzie idzie złotówka
# --------------------------------------------------------------------------- #
with tab_crude:
    st.subheader("Ile w cenie to surowiec, a ile podatki i marża")
    st.caption("Brent przeliczony na PLN za litr (1 baryłka = 159 l) na tle ceny "
               "detalicznej. Różnica to akcyza, opłata paliwowa, VAT, marża i przerób.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df[focus], name=f"{FUEL_LABEL[focus]} (detal)",
                             line=dict(color=COL[focus], width=2.2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["brent_pln_l"], name="Surowiec (Brent w PLN/l)",
                             line=dict(color=COL["brent"], width=1.6), fill="tozeroy",
                             fillcolor="rgba(154,160,171,.12)"))
    fig.update_yaxes(title="zł / litr")
    st.plotly_chart(style_axes(fig), use_container_width=True)

    wedge = (df_all[focus] - df_all["brent_pln_l"]).mean()
    crude_share = (df_all["brent_pln_l"] / df_all[focus]).mean() * 100
    a, b = st.columns(2)
    a.metric("Średnia 'narzut' ponad surowiec", f"{wedge:.2f} zł/l",
             help="Podatki + opłaty + marża + przerób")
    b.metric("Udział surowca w cenie", f"{crude_share:.0f}%")

# --------------------------------------------------------------------------- #
#  3. Sezonowość
# --------------------------------------------------------------------------- #
with tab_season:
    st.subheader(f"Sezonowość — {FUEL_LABEL[focus]}")
    tmp = df_all.copy()
    tmp["month"] = tmp["date"].dt.month
    tmp["year"] = tmp["date"].dt.year
    # odchylenie od średniej rocznej, by usunąć wpływ trendu
    tmp["dev"] = tmp.groupby("year")[focus].transform(lambda s: s - s.mean())
    monthly = tmp.groupby("month")["dev"].mean()
    months_pl = ["Sty", "Lut", "Mar", "Kwi", "Maj", "Cze",
                 "Lip", "Sie", "Wrz", "Paź", "Lis", "Gru"]
    fig = go.Figure(go.Bar(
        x=months_pl, y=monthly.values,
        marker_color=[COL["accent"] if v >= 0 else COL["ON"] for v in monthly.values],
    ))
    fig.update_yaxes(title="odchylenie od średniej rocznej (zł/l)")
    st.plotly_chart(style_axes(fig), use_container_width=True)
    peak = months_pl[int(monthly.idxmax()) - 1]
    low = months_pl[int(monthly.idxmin()) - 1]
    st.markdown(
        f"<div class='insight'>Po usunięciu trendu widać powtarzalny wzorzec: "
        f"najdrożej zwykle w okolicach <b>{peak}</b>, najtaniej w <b>{low}</b>.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Dekompozycja szeregu (trend / sezon / reszta)"):
        s = df_all.set_index("date")[focus].asfreq("D").interpolate()
        dec = seasonal_decompose(s, period=365, model="additive", extrapolate_trend="freq")
        for comp, name, color in [(dec.trend, "Trend", COL[focus]),
                                  (dec.seasonal, "Składnik sezonowy", COL["accent"]),
                                  (dec.resid, "Reszta", COL["muted"])]:
            f2 = go.Figure(go.Scatter(x=comp.index, y=comp.values,
                                      line=dict(color=color, width=1.4)))
            f2.update_layout(title=name, height=220)
            st.plotly_chart(style_axes(f2), use_container_width=True)

# --------------------------------------------------------------------------- #
#  4. Opóźnienie reakcji
# --------------------------------------------------------------------------- #
with tab_lag:
    st.subheader(f"Z jakim opóźnieniem stacje reagują na rynek — {FUEL_LABEL[focus]}")
    st.caption("Korelacja dziennej zmiany ceny paliwa ze zmianą ceny surowca sprzed "
               "N dni. Szczyt wskazuje typowe opóźnienie reakcji stacji.")
    cc = A.lag_correlation(df_all, focus, max_lag=30)
    lag = int(cc.loc[cc["corr"].idxmax(), "lag"])
    fig = go.Figure(go.Bar(
        x=cc["lag"], y=cc["corr"],
        marker_color=[COL["accent"] if l == lag else "rgba(245,166,35,.35)"
                      for l in cc["lag"]],
    ))
    fig.update_xaxes(title="opóźnienie (dni)")
    fig.update_yaxes(title="korelacja zmian")
    st.plotly_chart(style_axes(fig), use_container_width=True)
    st.markdown(
        f"<div class='insight'>Ceny na stacjach najsilniej odzwierciedlają rynek ropy "
        f"z opóźnieniem około <b>{lag} dni</b>. To opóźnienie wykorzystujemy w modelu "
        f"prognostycznym.</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
#  5. Asymetria — rakieta i piórko
# --------------------------------------------------------------------------- #
with tab_asym:
    st.subheader("Efekt „rakiety i piórka”")
    st.caption("Czy stacje szybciej podnoszą ceny przy drożejącej ropie, niż obniżają "
               "przy taniejącej? Porównujemy siłę reakcji na wzrosty i spadki surowca.")
    lag = A.best_lag(df_all, focus)
    asym = A.asymmetry(df_all, focus, lag)
    fig = go.Figure(go.Bar(
        x=["Reakcja na WZROST ropy", "Reakcja na SPADEK ropy"],
        y=[asym["beta_up"], asym["beta_down"]],
        marker_color=[COL["up"], COL["down"]],
        text=[f"{asym['beta_up']:.3f}", f"{asym['beta_down']:.3f}"],
        textposition="outside",
    ))
    fig.update_yaxes(title="siła przełożenia na cenę")
    st.plotly_chart(style_axes(fig), use_container_width=True)
    if asym["asymmetric"]:
        st.markdown(
            f"<div class='insight'>Potwierdzony efekt <b>rakiety i piórka</b>: przełożenie "
            f"wzrostów ropy na cenę jest ok. <b>{asym['ratio']:.1f}×</b> silniejsze niż "
            f"przełożenie spadków. Mówiąc prościej — drożeje szybko, tanieje powoli.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='insight'>W tym zakresie danych reakcja na wzrosty i "
                    "spadki ropy jest zbliżona.</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  6. Prognoza
# --------------------------------------------------------------------------- #
with tab_fc:
    st.subheader(f"Prognoza ceny — {FUEL_LABEL[focus]} (+{horizon} dni)")
    st.caption("Model: cena = f(surowiec sprzed N dni, kurs USD/PLN, sezonowość). "
               "Prognoza to scenariusz przy założeniu stałych Brent i kursu.")
    lag = A.best_lag(df_all, focus)
    fc = A.forecast(df_all, focus, lag, horizon=horizon)
    hist = fc[fc.kind == "historia"]
    fut = fc[fc.kind == "prognoza"]
    show_hist = hist[hist["date"] >= hist["date"].max() - pd.Timedelta(days=365)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(fut["date"]) + list(fut["date"][::-1]),
        y=list(fut["hi"]) + list(fut["lo"][::-1]),
        fill="toself", fillcolor="rgba(245,166,35,.15)",
        line=dict(width=0), name="przedział 95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=show_hist["date"], y=show_hist["value"],
                             name="historia", line=dict(color=COL[focus], width=2)))
    fig.add_trace(go.Scatter(x=fut["date"], y=fut["value"], name="prognoza",
                             line=dict(color=COL["accent"], width=2.4, dash="dot")))
    fig.update_yaxes(title="zł / litr")
    st.plotly_chart(style_axes(fig), use_container_width=True)

    a, b = st.columns(2)
    a.metric(f"Prognoza za {horizon} dni", f"{fut['value'].iloc[-1]:.2f} zł/l",
             f"{fut['value'].iloc[-1] - latest[focus]:+.2f} zł", delta_color="inverse")
    b.metric("Dopasowanie modelu (R²)", f"{fc.attrs['r2']:.3f}")
    st.caption("⚠️ Prognoza poglądowa — nie uwzględnia szoków geopolitycznych ani zmian "
               "podatkowych. Służy demonstracji metody, nie rekomendacji.")

st.markdown("---")
st.markdown(
    "<span class='src'>Projekt zaliczeniowy · analiza danych + komponent predykcyjny. "
    "Dane: paliwo.today, NBP, Yahoo Finance. Metodyka: cross-correlation, regresja "
    "asymetryczna, dekompozycja sezonowa, regresja liniowa z opóźnieniem.</span>",
    unsafe_allow_html=True,
)
