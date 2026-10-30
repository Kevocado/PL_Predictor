"""Kalshi fee formula (verbatim copy from predictor-hub services/paper/paper/fees.py)."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


TAKER_RATE = 0.07
MAKER_SHARE = 0.25


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _whole_price_cents(price_cents: float) -> int:
    value = Decimal(str(price_cents))
    rounded = value.to_integral_value(rounding=ROUND_HALF_UP)
    return max(0, min(100, int(rounded)))


def kalshi_fee_cents(price_cents: float, *, contracts: int = 1, maker: bool = False) -> float:
    if contracts <= 0:
        return 0.0
    p = _whole_price_cents(price_cents)
    numerator = 7 * contracts * p * (100 - p)
    denominator = 40000 if maker else 10000
    return float(_ceil_div(numerator, denominator))


def net_edge_pct(model_prob_pct: float, kalshi_price_cents: float, *, contracts: int = 1, maker: bool = False) -> float:
    gross_edge_pct = model_prob_pct - kalshi_price_cents
    fee_cents_total = kalshi_fee_cents(kalshi_price_cents, contracts=contracts, maker=maker)
    fee_pct_per_contract = fee_cents_total / contracts
    return gross_edge_pct - fee_pct_per_contract
