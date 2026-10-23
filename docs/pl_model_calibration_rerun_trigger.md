# PL-99 Data integrity — exact predicate (2026-10-23)

Doc: docs/pl_model_calibration_rerun_trigger.md (must state exact predicate).
Script fix: scripts/pl_model_calibration.py.

Predicate for n finished fixtures with COMPLETE model probabilities:
- `fixture_detail_by_event_id[eid]` exists (O/U + BTTS source)
- `f.get("finished") == True` (fixture finished)
- `predicted_home_win`, `predicted_draw`, `predicted_away_win` non-null (from fixture record, NOT fixture_detail — 1X2 lives in fixtures_by_gameweek fixture)
- `fixture_detail[eid]["over_2_5"]["prob"]`, `fixture_detail[eid]["under_2_5"]["prob"]`, `fixture_detail[eid]["btts_yes_prob"]` non-null (from fixture_detail)
- All six prob values are float (not NaN)
- 1X2 sum in [0.99, 1.01] per fixture (not just aggregate)

Previous bug: script counted every finished fixture with a detail record
and did not verify probabilities non-null; read 1X2 from fixture record
without null-check. Fixed.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
🤖 Generated with [Claude Code](https://claude.com/claude-code)
