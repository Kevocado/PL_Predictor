# PL value-bet redesign

## Diagnosis
Current flags compare model probability against de-vigged bookmaker odds (Shin) for 1X2, totals 2.5 and BTTS only; corners/cards have no live market. Threshold 4 percentage points net (after shrinkage); min probability 0.30. Cards/corners are not flagged as value bets.

## Calibration
All main markets (match result, totals 2.5, BTTS) treated as calibrated for gating; corners/cards not. n=70 this season, intervals wide (inconclusive); see docs/pl_model_calibration.md.

## Evaluation
Walk-forward not completed in this environment; ship rule keeps new logic active but evidence pending. CLV/profit interval not computed.
