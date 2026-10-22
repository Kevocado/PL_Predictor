"""The BTTS ladder is collected like the other PL markets.

Kalshi lists `KXEPLBTTS` — "Both Teams To Score" — as its own series, one
binary market per fixture, and the paper ledger needs its prices to pair with
PL's published `btts_yes_prob`. Without it the pl:btts segment has a model
probability and no price.

The series was checked against Kalshi's public series list rather than assumed:
`KXEPLBTTS` ("EPL Both Teams to Score") is listed and has open markets.
"""
from datetime import datetime, timezone

from pl_predictor.data import kalshi_lines as kl

BTTS = {
    "ticker": "KXEPLBTTS-26OCT25MUNBOU-BTTS", "event_ticker": "KXEPLBTTS-26OCT25MUNBOU",
    "title": "Both Teams To Score", "floor_strike": None, "status": "active", "result": "",
    "yes_bid_dollars": "0.5000", "yes_ask_dollars": "0.5300",
    "no_bid_dollars": "0.4700", "no_ask_dollars": "0.5000",
    "last_price_dollars": "0.5100", "volume_fp": "12.5", "open_interest_fp": "12.5",
    "occurrence_datetime": "2026-10-25T18:00:00Z", "close_time": "2026-10-27T14:00:00Z",
    "expiration_value": "",
}


def test_btts_is_a_collected_series():
    assert "KXEPLBTTS" in kl.SERIES


def test_btts_normalises_as_its_own_market():
    row = kl.normalize(BTTS, "KXEPLBTTS", "2026-10-24T03:41:33+00:00")
    assert row["stat"] == "btts"
    assert row["series"] == "KXEPLBTTS"
    assert row["event_ticker"] == "KXEPLBTTS-26OCT25MUNBOU"
    assert row["ticker"] == "KXEPLBTTS-26OCT25MUNBOU-BTTS"
    assert row["line"] is None          # a binary market, not a ladder rung
    assert (row["yes_bid"], row["yes_ask"]) == (0.50, 0.53)
    assert (row["no_bid"], row["no_ask"]) == (0.47, 0.50)
    # occurrence is the kickoff, not Kalshi's much later close_time.
    assert row["occurrence"] == "2026-10-25T18:00:00Z"
    assert row["close_time"] == "2026-10-27T14:00:00Z"


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b

    def raise_for_status(self):
        assert self.status_code < 400


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        market = BTTS if params["series_ticker"] == "KXEPLBTTS" else {"ticker": "x"}
        return _Resp({"markets": [market], "cursor": ""})


def test_collect_asks_for_the_btts_series():
    s = _Session()
    kl.collect(s, now=datetime(2026, 10, 24, 3, tzinfo=timezone.utc))
    asked = {(c["series_ticker"], c["status"]) for c in s.calls}
    assert ("KXEPLBTTS", "open") in asked
    assert ("KXEPLBTTS", "settled") in asked


def test_a_collected_btts_row_carries_its_own_stat():
    s = _Session()
    rows = kl.collect(s, now=datetime(2026, 10, 24, 3, tzinfo=timezone.utc))
    btts = [r for r in rows if r["series"] == "KXEPLBTTS"]
    assert btts and all(r["stat"] == "btts" for r in btts)
