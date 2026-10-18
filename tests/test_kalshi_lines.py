import gzip
import json
from datetime import datetime, timezone

from pl_predictor.data import kalshi_lines as kl

LFC = {
    "ticker": "KXEPLGAME-26NOV01LFCARS-LFC", "event_ticker": "KXEPLGAME-26NOV01LFCARS",
    "title": "Liverpool wins", "floor_strike": None, "status": "active", "result": "",
    "yes_bid_dollars": "0.2600", "yes_ask_dollars": "0.3300",
    "no_bid_dollars": "0.6700", "no_ask_dollars": "0.7400",
    "last_price_dollars": "0.0000", "volume_fp": "1.23", "open_interest_fp": "1.23",
    "occurrence_datetime": "2026-11-01T19:30:00Z", "close_time": "2026-11-03T16:30:00Z",
    "expiration_value": "",
}


def test_normalize_keeps_the_quotes_as_probabilities_and_occurrence_as_kickoff():
    row = kl.normalize(LFC, "KXEPLGAME", "2026-10-18T03:41:33+00:00")
    assert row["stat"] == "moneyline"
    assert (row["yes_bid"], row["yes_ask"]) == (0.26, 0.33)
    assert (row["no_bid"], row["no_ask"]) == (0.67, 0.74)
    assert row["occurrence"] == "2026-11-01T19:30:00Z"
    assert row["close_time"] == "2026-11-03T16:30:00Z"
    assert row["result"] is None and row["status"] == "active"


def test_a_settled_market_carries_its_outcome():
    row = kl.normalize({**LFC, "status": "finalized", "result": "yes",
                       "expiration_value": "1"}, "KXEPLGAME", "t")
    assert row["result"] == "yes" and row["settlement_value"] == "1"


def test_missing_prices_are_none_not_zero():
    row = kl.normalize({"title": "Tie is the result", "floor_strike": None}, "KXEPLGAME", "t")
    assert row["yes_bid"] is None and row["yes_ask"] is None


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
        if params.get("cursor") == "p2":
            return _Resp({"markets": [LFC], "cursor": ""})
        return _Resp({"markets": [LFC], "cursor": "p2" if params["series_ticker"] == "KXEPLGAME" and params["status"] == "open" else ""})


def test_collect_pages_through_cursors_and_asks_for_open_and_recently_settled():
    s = _Session()
    rows = kl.collect(s, now=datetime(2026, 10, 18, 3, tzinfo=timezone.utc))
    statuses = {(c["series_ticker"], c["status"]) for c in s.calls}
    assert ("KXEPLGAME", "open") in statuses and ("KXEPLTOTAL", "settled") in statuses
    assert any(c.get("cursor") == "p2" for c in s.calls)
    assert len(rows) == 2 * 2 + 1                       # 2 series x (open, settled) + the extra page
    assert all(r["snapshot_ts"].startswith("2026-10-18") for r in rows)


def test_write_snapshot_appends_one_gzip_file_per_utc_day(tmp_path):
    rows = [kl.normalize(LFC, "KXEPLGAME", "t")]
    now = datetime(2026, 10, 18, 3, tzinfo=timezone.utc)
    p1 = kl.write_snapshot(rows, tmp_path, now)
    p2 = kl.write_snapshot(rows, tmp_path, now)
    assert p1 == p2 and p1.name == "kalshi_pl_lines_2026-10-18.jsonl.gz"
    lines = gzip.open(p1, "rt").read().splitlines()
    assert len(lines) == 2
    assert kl.write_snapshot([], tmp_path, now) is None
