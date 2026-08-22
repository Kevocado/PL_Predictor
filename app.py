"""
app.py — PL Predictor dashboard.

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pl_predictor.config import ODDS_API_KEY
from pl_predictor.data.fixtures import get_upcoming_fixtures
from pl_predictor.data.odds_api import OddsAPIKeyMissing, fetch_epl_odds
from pl_predictor.evaluate import backtest as backtest_lib
from pl_predictor.evaluate import calibration
from pl_predictor.features.build import build_features_for_fixtures, build_training_frame
from pl_predictor.models import manifest as manifest_lib
from pl_predictor.models import scoreline
from pl_predictor.models.manifest import chronological_split
from pl_predictor.odds import value_bets

st.set_page_config(page_title="PL Predictor", page_icon="⚽", layout="wide")


@st.cache_data(ttl=3600)
def get_manifest():
    return manifest_lib.load_manifest()


@st.cache_resource(ttl=3600)
def get_models():
    return manifest_lib.load_models()


@st.cache_data(ttl=1800, show_spinner="Fetching upcoming fixtures...")
def get_fixtures():
    return get_upcoming_fixtures()


@st.cache_data(ttl=1800, show_spinner="Fetching live odds...")
def get_odds():
    try:
        return fetch_epl_odds()
    except OddsAPIKeyMissing:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner="Scoring upcoming fixtures...")
def get_predictions():
    fixtures = get_fixtures()
    models = get_models()
    if fixtures.empty:
        return pd.DataFrame(), fixtures
    odds_df = get_odds()
    table = value_bets.build_value_bet_table(fixtures, odds_df, models)
    return table, fixtures


def models_available() -> bool:
    return manifest_lib.MANIFEST_PATH.exists()


def render_sidebar():
    st.sidebar.title("⚽ PL Predictor")
    st.sidebar.caption("Match outcomes, scorelines, and betting markets.")

    if not ODDS_API_KEY:
        st.sidebar.warning(
            "No `ODDS_API_KEY` set — showing model predictions only, no live market "
            "comparison. Get a free key at [the-odds-api.com](https://the-odds-api.com/) "
            "and add `ODDS_API_KEY=...` to your `.env` file."
        )

    if st.sidebar.button("\U0001f504 Refresh fixtures"):
        get_fixtures.clear()
        get_predictions.clear()
        st.rerun()

    if st.sidebar.button("\U0001f4b0 Refresh odds"):
        get_odds.clear()
        get_predictions.clear()
        st.rerun()

    if st.sidebar.button("\U0001f3cb️ Retrain models"):
        with st.status("Training models on historical seasons...", expanded=True) as status:
            manifest = manifest_lib.train_all()
            st.write(f"Trained on {manifest['n_train']} matches, validated on {manifest['n_val']}.")
            st.write(f"Chosen scoreline model: {manifest['scoreline']['chosen_model']}")
            status.update(label="Done", state="complete")
        get_manifest.clear()
        get_models.clear()
        get_predictions.clear()
        st.rerun()

    if models_available():
        manifest = get_manifest()
        st.sidebar.caption(f"Models trained: {manifest['trained_at'][:19]} UTC")
        st.sidebar.caption(f"Seasons: {manifest['seasons'][0]} – {manifest['seasons'][-1]}")


def render_fixtures_tab():
    table, fixtures = get_predictions()
    if table.empty:
        st.info("No upcoming fixtures found.")
        return

    display = table.copy()
    for col in ["home_win_prob", "draw_prob", "away_win_prob", "btts_yes_prob", "over_2_5_prob"]:
        display[col] = (display[col] * 100).round(1)
    display["value_bets"] = display["value_bet_flags"].apply(lambda flags: ", ".join(flags) if flags else "")

    st.dataframe(
        display[
            [
                "commence_time", "team_home", "team_away", "home_win_prob", "draw_prob", "away_win_prob",
                "top_scoreline", "btts_yes_prob", "over_2_5_prob", "value_bets", "is_fallback_prediction",
            ]
        ].rename(
            columns={
                "commence_time": "Kickoff", "team_home": "Home", "team_away": "Away",
                "home_win_prob": "Home %", "draw_prob": "Draw %", "away_win_prob": "Away %",
                "top_scoreline": "Likely score", "btts_yes_prob": "BTTS %", "over_2_5_prob": "O2.5 %",
                "value_bets": "Value bets", "is_fallback_prediction": "New-team fallback",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=500,
    )
    st.caption(
        "Corners and cards predictions are model-only for now — The Odds API's free markets "
        "don't cover them, so there's no live line to compare against yet."
    )


def render_scoreline_tab():
    table, fixtures = get_predictions()
    if table.empty:
        st.info("No upcoming fixtures found.")
        return

    models = get_models()
    options = [f"{r.team_home} vs {r.team_away}" for r in table.itertuples()]
    choice = st.selectbox("Fixture", options)
    idx = options.index(choice)
    row = table.iloc[idx]

    pred = scoreline.predict_fixture(models["scoreline"], row["team_home"], row["team_away"])
    grid = pred["grid"][:6, :6]
    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            x=[str(i) for i in range(6)],
            y=[str(i) for i in range(6)],
            colorscale="Blues",
            text=[[f"{v:.1%}" for v in row_] for row_ in grid],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=f"{row['team_home']} (home goals) vs {row['team_away']} (away goals)",
        xaxis_title=f"{row['team_away']} goals",
        yaxis_title=f"{row['team_home']} goals",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{row['team_home']} win", f"{pred['home_win']:.1%}")
    c2.metric("Draw", f"{pred['draw']:.1%}")
    c3.metric(f"{row['team_away']} win", f"{pred['away_win']:.1%}")

    feat_row = build_features_for_fixtures(
        pd.DataFrame([{"team_home": row["team_home"], "team_away": row["team_away"]}])
    ).iloc[0]
    market_preds = value_bets.predict_market_models_for_fixture(models, feat_row)
    c4, c5 = st.columns(2)
    c4.metric("Expected corners", f"{market_preds['corners']['lambda']:.1f}")
    c5.metric("Expected cards", f"{market_preds['cards']['lambda']:.1f}")


def render_calibration_tab():
    manifest = get_manifest()
    models = get_models()

    st.subheader("Forecast calibration (held-out season)")
    scoreline_metrics = manifest["scoreline"][manifest["scoreline"]["chosen_model"]]["metrics"]
    st.write(
        f"**Model** — RPS: {scoreline_metrics['rps']:.4f} · Brier: {scoreline_metrics['brier']:.4f} · "
        f"fallback rate (new teams): {scoreline_metrics['fallback_rate']:.1%}"
    )
    st.caption(
        "Lower is better for RPS/Brier. Compare against the de-vigged closing-odds baseline "
        "below — a well-calibrated model should be close to, not dramatically worse than, the market."
    )

    if st.button("Run calibration + backtest against held-out season"):
        with st.status("Building training frame and evaluating...", expanded=True) as status:
            df, feature_cols = build_training_frame()
            train_df, val_df = chronological_split(df)
            bookmaker = calibration.bookmaker_calibration(val_df)
            naive = calibration.naive_favourite_baseline(val_df)
            st.write(f"Model RPS: {scoreline_metrics['rps']:.4f}")
            if bookmaker:
                st.write(f"Bookmaker (closing odds, de-vigged) RPS: {bookmaker['rps']:.4f}")
            st.write(f"Naive baseline RPS: {naive['rps']:.4f}")

            start, end = str(val_df["date"].min().date()), str(val_df["date"].max().date())
            results = backtest_lib.run_value_bet_backtest(val_df, models["scoreline"], start, end)
            status.update(label="Done", state="complete")
        st.subheader("Value-bet backtest (edge > 5%, held-out season)")
        st.json(results)
        st.caption(
            "A strongly positive ROI here is a red flag for overfitting, not a sign the "
            "model has found a real edge — expect roughly break-even to slightly negative."
        )

    st.subheader("Corners / cards model")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Corners**")
        st.json(manifest["corners"]["metrics"])
    with c2:
        st.write("**Cards**")
        st.json(manifest["cards"]["metrics"])


def render_value_bets_tab():
    table, fixtures = get_predictions()
    if table.empty or "home_win_edge" not in table.columns or table["home_win_edge"].isna().all():
        st.info("No live odds available — set ODDS_API_KEY to see value bets.")
        return

    flagged = table[table["value_bet_flags"].apply(len) > 0].copy()
    if flagged.empty:
        st.info("No value bets above the edge threshold right now.")
        return

    rows = []
    for _, r in flagged.iterrows():
        for side in r["value_bet_flags"]:
            rows.append(
                {
                    "Fixture": f"{r['team_home']} vs {r['team_away']}",
                    "Side": side,
                    "Model prob": f"{r[f'{side}_prob']:.1%}",
                    "Market implied": f"{r[f'{side}_implied']:.1%}",
                    "Edge": f"{r[f'{side}_edge']:.1%}",
                }
            )
    st.dataframe(pd.DataFrame(rows).sort_values("Edge", ascending=False), use_container_width=True, hide_index=True)


def main():
    render_sidebar()

    if not models_available():
        st.title("⚽ PL Predictor")
        st.warning(
            "No trained models yet. Click **Retrain models** in the sidebar "
            "(or run `python -m pl_predictor.models.manifest`) to get started."
        )
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["\U0001f4c5 Upcoming Fixtures", "\U0001f3af Scoreline Grid", "\U0001f4ca Calibration & Backtest", "\U0001f4b8 Value Bets"]
    )
    with tab1:
        render_fixtures_tab()
    with tab2:
        render_scoreline_tab()
    with tab3:
        render_calibration_tab()
    with tab4:
        render_value_bets_tab()


if __name__ == "__main__":
    main()
