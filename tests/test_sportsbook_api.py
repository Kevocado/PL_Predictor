import pandas as pd

from pl_predictor.data.sportsbook_api import event_to_rows, match_project_event_ids

# Trimmed from a real GET /v0/competitions/{key}/events response
# (2026-09-11) -- confirmed live: some duplicate MONEYLINE_3WAY market
# entries carry no books at all, KALSHI mixes WIN_NO/DRAW_NO (lay) rows
# into the same MONEYLINE_3WAY outcomes list, and totals lines vary by book
# (2.5 and 3.0 both seen for the same fixture).
_LIST_EVENTS = [
    {
        "key": "qyLO-Ajup-5zBW",
        "name": "Manchester United @ Fulham",
        "startTime": "2026-09-20T15:30:00.000Z",
        "homeParticipantKey": "jFtf-wjcs-Wz8k",
        "participants": [
            {"key": "jFtf-wjcs-Wz8k", "name": "Fulham"},
            {"key": "oAL2-wjcs-9vzu", "name": "Manchester United"},
        ],
    }
]

_EVENT_ODDS = {
    "key": "qyLO-Ajup-5zBW",
    "homeParticipantKey": "jFtf-wjcs-Wz8k",
    "markets": [
        {
            "type": "MONEYLINE_3WAY",
            "segment": "REGULATION_TIME",
            "outcomes": {
                "BET_PARX": [
                    {"type": "WIN", "participantKey": "jFtf-wjcs-Wz8k", "payout": 3.45},
                    {"type": "WIN", "participantKey": "oAL2-wjcs-9vzu", "payout": 1.94},
                    {"type": "DRAW", "participantKey": None, "payout": 3.75},
                ],
                "KALSHI": [
                    {"type": "WIN", "participantKey": "jFtf-wjcs-Wz8k", "payout": 3.70},
                    {"type": "WIN", "participantKey": "oAL2-wjcs-9vzu", "payout": 2.04},
                    {"type": "DRAW", "participantKey": None, "payout": 3.85},
                    {"type": "WIN_NO", "participantKey": "jFtf-wjcs-Wz8k", "payout": 1.35},
                    {"type": "DRAW_NO", "participantKey": None, "payout": 1.32},
                ],
            },
        },
        # Stale duplicate market key with no books at all -- confirmed live.
        {"type": "MONEYLINE_3WAY", "segment": "REGULATION_TIME", "outcomes": {}},
        {
            "type": "POINT_TOTAL",
            "segment": "REGULATION_TIME",
            "outcomes": {
                "BET_PARX": [
                    {"type": "OVER", "modifier": 2.5, "payout": 1.61},
                    {"type": "UNDER", "modifier": 2.5, "payout": 2.30},
                ],
                "BOVADA": [
                    # Different line at this book -- must not blend into the 2.5 line.
                    {"type": "OVER", "modifier": 3.0, "payout": 1.91},
                    {"type": "UNDER", "modifier": 3.0, "payout": 1.91},
                ],
            },
        },
        {
            "type": "BOTH_TEAMS_TO_SCORE",
            "segment": "REGULATION_TIME",
            "outcomes": {
                "BET_PARX": [
                    {"type": "YES", "modifier": 0, "payout": 1.53},
                    {"type": "NO", "modifier": 0, "payout": 2.35},
                ],
            },
        },
        # Different segment -- must be ignored entirely.
        {
            "type": "MONEYLINE_3WAY",
            "segment": "HALF_1",
            "outcomes": {"BET_PARX": [{"type": "WIN", "participantKey": "jFtf-wjcs-Wz8k", "payout": 1.9}]},
        },
    ],
}


def test_match_project_event_ids_matches_by_team_names_and_time():
    fixtures_df = pd.DataFrame(
        [{"event_id": "proj-1", "team_home": "Fulham", "team_away": "Man United", "commence_time": "2026-09-20T15:30:00Z"}]
    )
    mapping = match_project_event_ids(fixtures_df, _LIST_EVENTS)
    assert mapping == {"proj-1": "qyLO-Ajup-5zBW"}


def test_match_project_event_ids_skips_unmatched_fixture():
    fixtures_df = pd.DataFrame(
        [{"event_id": "proj-2", "team_home": "Arsenal", "team_away": "Chelsea", "commence_time": "2026-09-20T15:30:00Z"}]
    )
    mapping = match_project_event_ids(fixtures_df, _LIST_EVENTS)
    assert mapping == {}


def test_event_to_rows_h2h_splits_home_away_by_participant_key():
    rows = event_to_rows(
        _EVENT_ODDS, project_event_id="proj-1", team_home="Fulham", team_away="Man United",
        commence_time="2026-09-20T15:30:00Z", fetched_at=pd.Timestamp("2026-09-11T00:00:00Z"),
    )
    h2h = [r for r in rows if r["market"] == "h2h"]
    assert {r["outcome_name"] for r in h2h} == {"Fulham", "Man United", "Draw"}
    fulham_prices = sorted(r["price"] for r in h2h if r["outcome_name"] == "Fulham")
    assert fulham_prices == [3.45, 3.70]


def test_event_to_rows_h2h_excludes_kalshi_no_side():
    rows = event_to_rows(
        _EVENT_ODDS, project_event_id="proj-1", team_home="Fulham", team_away="Man United",
        commence_time="2026-09-20T15:30:00Z", fetched_at=pd.Timestamp("2026-09-11T00:00:00Z"),
    )
    prices = [r["price"] for r in rows if r["market"] == "h2h"]
    assert 1.35 not in prices and 1.32 not in prices


def test_event_to_rows_totals_keeps_only_2_5_line():
    rows = event_to_rows(
        _EVENT_ODDS, project_event_id="proj-1", team_home="Fulham", team_away="Man United",
        commence_time="2026-09-20T15:30:00Z", fetched_at=pd.Timestamp("2026-09-11T00:00:00Z"),
    )
    totals = [r for r in rows if r["market"] == "totals"]
    assert all(r["point"] == 2.5 for r in totals)
    assert {r["price"] for r in totals} == {1.61, 2.30}


def test_event_to_rows_includes_btts():
    rows = event_to_rows(
        _EVENT_ODDS, project_event_id="proj-1", team_home="Fulham", team_away="Man United",
        commence_time="2026-09-20T15:30:00Z", fetched_at=pd.Timestamp("2026-09-11T00:00:00Z"),
    )
    btts = {r["outcome_name"]: r["price"] for r in rows if r["market"] == "btts"}
    assert btts == {"Yes": 1.53, "No": 2.35}
