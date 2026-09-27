"""Leakage and correctness tests for opponent-defensive-strength features.

The player-rate features in `player_form` are entirely opponent-blind: a
player who scores four against a relegated side is recorded exactly like one
who scores four against the league leaders. These tests pin the feature that
fixes that, and — more importantly — pin that it uses only prior matches.
"""

import pandas as pd
import pytest

from pl_predictor.data.fpl_history import default_completed_seasons, load_player_gw_history
from pl_predictor.features.opponent_defence import (
    build_opponent_defence_features,
    build_team_match_history,
    rate_opponents,
    rate_team_matches,
    solve_team_id_names,
    team_matches_from_matches_df,
)

KICKOFF_ORIGIN = pd.Timestamp("2024-08-17T14:00")

# Recovering an `opponent_team` id to a club name relies on every club having met
# every other, so the synthetic seasons below are complete round-robins. A
# two-team fixture is not enough: the map cannot resolve, every row comes back
# NaN, and a leakage test over all-NaN asserts nothing at all.
TEAMS = ["Leaky", "Tight", "Fulham", "Spurs", "Everton"]
ATTACK = {"Leaky": 0, "Tight": 2, "Fulham": 1, "Spurs": 1, "Everton": 1}
LEAK = {"Leaky": 4, "Tight": 0, "Fulham": 1, "Spurs": 1, "Everton": 1}


def _history(records: list[dict]) -> pd.DataFrame:
    """Minimal player-gameweek frame carrying only what the feature needs."""
    frame = pd.DataFrame(records)
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"])
    return frame


def _team_id(name: str) -> str:
    return str(TEAMS.index(name) + 1)


def _round_robin(season: str = "2024-25", attack=None, leak=None) -> pd.DataFrame:
    """One complete round-robin, so the id map resolves for every club."""
    attack = attack or ATTACK
    leak = leak or LEAK
    pairs = [(a, b) for i, a in enumerate(TEAMS) for b in TEAMS[i + 1:]]
    rows = []
    for slot, (home, away) in enumerate(pairs):
        kickoff = KICKOFF_ORIGIN + pd.Timedelta(days=7 * slot)
        home_goals, away_goals = attack[home] + leak[away], attack[away] + leak[home]
        for team, opponent, was_home in ((home, away, "True"), (away, home, "False")):
            rows.append({"season": season, "team": team, "opponent_team": _team_id(opponent),
                         "kickoff_time": kickoff, "was_home": was_home,
                         "team_h_score": home_goals, "team_a_score": away_goals})
    return _history(rows)


def _fixture(team: str, opponent: str, day_offset: int, home_goals: int = 1, away_goals: int = 0) -> pd.DataFrame:
    """A complete fixture, as the archive always records both sides.

    Both sides matter: the rating is looked up on the *opponent's* row at that
    kickoff, so a one-sided fixture would silently resolve to nothing.
    """
    kickoff = KICKOFF_ORIGIN + pd.Timedelta(days=7 * day_offset)
    return _history([
        {"season": "2024-25", "team": team, "opponent_team": _team_id(opponent), "kickoff_time": kickoff,
         "was_home": "True", "team_h_score": home_goals, "team_a_score": away_goals},
        {"season": "2024-25", "team": opponent, "opponent_team": _team_id(team), "kickoff_time": kickoff,
         "was_home": "False", "team_h_score": home_goals, "team_a_score": away_goals},
    ])


def _rating(team: str, opponent: str, day_offset: int, *, extra: tuple = (), **score: int) -> float:
    """The `last3` rating a player gets for facing `opponent` on `day_offset`.

    `extra` holds whole further fixtures, so a test can add later results
    without shifting the row positions of the fixture being rated.
    """
    frame = pd.concat([_round_robin(), *extra, _fixture(team, opponent, day_offset, **score)], ignore_index=True)
    features, _ = build_opponent_defence_features(frame)
    kickoff = KICKOFF_ORIGIN + pd.Timedelta(days=7 * day_offset)
    selected = features.loc[(frame["team"] == team) & (frame["kickoff_time"] == kickoff)]
    return float(selected["opponent_defence_last3"].iloc[0])


def test_goals_are_attributed_to_the_side_that_played_them():
    """`team_h_score`/`team_a_score` are always from the fixture's own
    perspective, so the away side's goals are the *second* column. Getting
    this backwards would hand every player their opponent's goals."""
    history = _history([
        {"season": "2024-25", "team": "Home", "opponent_team": "9", "kickoff_time": "2024-08-17T14:00",
         "was_home": "True", "team_h_score": 3, "team_a_score": 0},
        {"season": "2024-25", "team": "Away", "opponent_team": "1", "kickoff_time": "2024-08-17T14:00",
         "was_home": "False", "team_h_score": 3, "team_a_score": 0},
    ])
    matches = build_team_match_history(history).set_index("team")
    assert (matches.loc["Home", "goals_for"], matches.loc["Home", "goals_against"]) == (3, 0)
    assert (matches.loc["Away", "goals_for"], matches.loc["Away", "goals_against"]) == (0, 3)


def test_repeated_player_rows_collapse_to_one_team_fixture():
    """A fixture has ~25 player rows. The team match history must be one row
    per team-fixture, or every team's rolling window counts a match twice."""
    history = _history([
        {"season": "2024-25", "team": "Home", "opponent_team": "9", "kickoff_time": "2024-08-17T14:00",
         "was_home": "True", "team_h_score": 1, "team_a_score": 1} for _ in range(25)
    ])
    assert len(build_team_match_history(history)) == 1


def test_the_opening_fixture_has_no_rating_but_the_rest_of_the_season_does():
    """A defence cannot be rated before it has played: both clubs in the
    season's opening fixture are debuting, so neither has a prior record. The
    frame builder then fills the gap with the league-average prior, exactly as
    for the other cold-start features. Asserted as a population, because an
    all-NaN frame would satisfy a weaker version of this."""
    season = _round_robin()
    features, _ = build_opponent_defence_features(season)
    kickoffs = season.drop_duplicates("kickoff_time")["kickoff_time"].sort_values()
    column = "opponent_defence_last3"
    opening = features.loc[season["kickoff_time"] == kickoffs.iloc[0], column]
    closing = features.loc[season["kickoff_time"] == kickoffs.iloc[-1], column]
    assert opening.isna().all(), "neither club has played before the opening fixture"
    assert closing.notna().all(), "every club has three prior matches by the last fixture"


def test_a_leaky_defence_scores_above_a_tight_one():
    """The whole point of the feature: the rating is the opponent's own goals
    conceded minus its own goals scored over prior matches. Positive means the
    player faces a defence that leaks more than it scores."""
    assert _rating("Fulham", "Leaky", 90) > 0
    assert _rating("Fulham", "Tight", 90) < 0


def test_the_rating_follows_the_opponent_not_the_players_own_team():
    """A regression guard for the join. If the rating were attached to the
    player's own club, Fulham's rating would be identical whoever it faced."""
    assert _rating("Fulham", "Leaky", 90) != _rating("Fulham", "Tight", 90)


def test_the_rating_ignores_the_match_it_describes():
    """Leakage guard, and the one most likely to be got wrong: a rating of 0-6
    or 9-0 in the very fixture being rated must not move the rating, because
    the rating describes prior matches only."""
    baseline = _rating("Fulham", "Leaky", 90)
    assert baseline == _rating("Fulham", "Leaky", 90, home_goals=0, away_goals=6)
    assert baseline == _rating("Fulham", "Leaky", 90, home_goals=7, away_goals=9)


def test_later_results_do_not_move_an_earlier_rating():
    """The complementary guard. Adding fixtures *after* the one being rated
    must leave that rating untouched; a missing `shift(1)` would let them in."""
    baseline = _rating("Fulham", "Leaky", 50)
    assert baseline == baseline, "the rated fixture must resolve to a real number"
    after = _rating("Fulham", "Leaky", 50, extra=(_fixture("Leaky", "Tight", 80, home_goals=0, away_goals=9),))
    assert baseline == after


def test_every_completed_season_resolves_to_a_full_twenty_club_map():
    """`opponent_team` is season-local, so a map must be rebuilt per season. If a
    season were silently skipped, every player facing a resolvable opponent
    would lose the feature and the model would fall back to the league prior
    without anyone noticing."""
    history = load_player_gw_history(seasons=default_completed_seasons())
    mapping = solve_team_id_names(history)
    assert set(mapping) == set(default_completed_seasons())
    for season, ids in mapping.items():
        assert len(ids) == 20, f"{season} expected 20 clubs, got {len(ids)}"
        assert len(set(ids.values())) == 20, f"{season} map is not one club per id"


def test_the_feature_is_actually_populated_on_real_history():
    """A join that silently returns all-NaN passes every leakage test above and
    ships a feature that does nothing. This is the same failure mode as the CFB
    team/player reconciliation, so it is asserted rather than assumed."""
    history = load_player_gw_history(seasons=default_completed_seasons())
    features, cols = build_opponent_defence_features(history)
    assert len(features) == len(history)
    assert features.index.equals(history.index), "features must align row-for-row with the input"
    populated = features[cols].notna().any(axis=1).mean()
    assert populated > 0.5, f"only {populated:.1%} of rows have any opponent rating"


def test_a_partly_played_season_is_dropped_even_when_every_id_resolves_once():
    """A season in progress can resolve every id to exactly one club and still be
    unusable, because two ids can land on the same club once a side has met
    everybody. The `never met` rule only identifies clubs in a complete
    round-robin, so a season is only accepted when the result is one club per
    id — otherwise a partial map looks correct and quietly mis-rates players."""
    history = _history([
        # Arsenal has met only id 3; Chelsea has met 1 and 2; the third club all.
        {"season": "2024-25", "team": "Arsenal", "opponent_team": "3", "kickoff_time": "2024-08-17T14:00", "was_home": "True", "team_h_score": 1, "team_a_score": 0},
        {"season": "2024-25", "team": "Chelsea", "opponent_team": "1", "kickoff_time": "2024-08-24T14:00", "was_home": "True", "team_h_score": 1, "team_a_score": 0},
        {"season": "2024-25", "team": "Chelsea", "opponent_team": "2", "kickoff_time": "2024-08-31T14:00", "was_home": "True", "team_h_score": 1, "team_a_score": 0},
        {"season": "2024-25", "team": "Everton", "opponent_team": "1", "kickoff_time": "2024-09-07T14:00", "was_home": "True", "team_h_score": 1, "team_a_score": 0},
        {"season": "2024-25", "team": "Everton", "opponent_team": "2", "kickoff_time": "2024-09-14T14:00", "was_home": "True", "team_h_score": 1, "team_a_score": 0},
        {"season": "2024-25", "team": "Everton", "opponent_team": "3", "kickoff_time": "2024-09-21T14:00", "was_home": "True", "team_h_score": 1, "team_a_score": 0},
    ])
    assert solve_team_id_names(history) == {}, "a non-bijective season must be dropped, not partly returned"


def test_a_season_the_archive_cannot_describe_is_skipped_not_guessed():
    """A lone club has no team it fails to meet, so the id map is unresolvable
    for that season. Returning a partial map would look identical to a correct
    one and quietly corrupt the feature, so the season is dropped whole."""
    history = _history([
        {"season": "2016-17", "team": "Arsenal", "opponent_team": "1", "kickoff_time": "2016-08-13T14:00",
         "was_home": "True", "team_h_score": 2, "team_a_score": 1},
    ])
    assert solve_team_id_names(history) == {}
    features, cols = build_opponent_defence_features(history)
    assert list(features.columns) == cols
    assert features.isna().all().all()


def test_a_frame_without_the_needed_columns_produces_nothing_rather_than_raising():
    """Seasons before the 2020 archive schema have no `opponent_team`. Callers
    concatenate seasons, so an old one must not take the whole build down."""
    legacy = pd.DataFrame({"name": ["A"], "goals_scored": [1]})
    assert build_team_match_history(legacy).empty
    assert solve_team_id_names(legacy) == {}
    features, cols = build_opponent_defence_features(legacy)
    assert list(features.columns) == cols
    assert len(features) == 1
    assert features.isna().all().all()


# --- serving adapter -------------------------------------------------------
#
# The rating is computed in exactly one place (`_rate_team_matches`) and reached
# from two directions: the FPL archive when fitting, and `to_team_perspective`
# when serving. The tests below are what make promoting that safe. A second
# implementation is not a style preference here -- it is how NFL ended up with
# two tests asserting 82 and 81.5 for the same player and the same week.


def _generic_team_matches(history: pd.DataFrame) -> pd.DataFrame:
    """The archive in the generic team-match shape `rate_team_matches` takes.

    The same shape `team_matches_from_matches_df` produces on the serving side.
    """
    matches = build_team_match_history(history)
    return matches.rename(columns={"kickoff_time": "date"})[
        ["season", "team", "date", "goals_for", "goals_against"]
    ]


def test_the_serving_path_agrees_with_the_training_path_on_synthetic_data():
    """Same numbers whether a rating is asked for as a player feature (fitting)
    or as a fixture feature (serving). With a shared `_rate_team_matches` this
    should hold exactly; if it ever stops holding, the two definitions have
    drifted and every number produced here is suspect."""
    history = _round_robin()
    team_matches = _generic_team_matches(history)
    combined = pd.concat([history, _fixture("Fulham", "Leaky", 90)], ignore_index=True)
    training, _ = build_opponent_defence_features(combined)

    trained = training.loc[(combined["team"] == "Fulham").values, "opponent_defence_last3"].iloc[-1]
    assert trained == trained, "sanity: the training path produced a real number"

    # The same fixture, asked for the other way round.
    fixtures = _history([{
        "season": "2024-25", "opponent": "Leaky",
        "kickoff_time": KICKOFF_ORIGIN + pd.Timedelta(days=7 * 90),
    }]).rename(columns={"kickoff_time": "date"})
    served = rate_opponents(team_matches, fixtures)["opponent_defence_last3"].iloc[0]
    assert served == pytest.approx(trained)


def test_a_clubs_first_upcoming_fixture_has_no_rating_through_the_serving_path():
    """The serving path must not invent a rating for a defence that has not
    played. Requesting one appends a placeholder row, and a placeholder with no
    prior matches has nothing behind it."""
    history = _round_robin()
    team_matches = _generic_team_matches(history)
    first_date = history["kickoff_time"].min()
    fixtures = _history([{"season": "2024-25", "opponent": "Leaky", "kickoff_time": first_date}]).rename(
        columns={"kickoff_time": "date"})
    rated = rate_opponents(team_matches, fixtures)["opponent_defence_last3"]
    assert rated.iloc[0] != rated.iloc[0], "Leaky's opening fixture must not be rated"


def test_a_placeholder_fixture_does_not_leak_into_later_ratings():
    """The serving path appends placeholder rows, so a later rating must not see
    them. If it did, asking about an upcoming match would change the answer to a
    question about a different upcoming match."""
    history = _round_robin()
    team_matches = _generic_team_matches(history)
    one = _history([{"season": "2024-25", "opponent": "Leaky",
                     "kickoff_time": KICKOFF_ORIGIN + pd.Timedelta(days=7 * 50)}]).rename(columns={"kickoff_time": "date"})
    two = pd.concat([one, _history([{"season": "2024-25", "opponent": "Leaky",
                                      "kickoff_time": KICKOFF_ORIGIN + pd.Timedelta(days=7 * 60)}]
                                    ).rename(columns={"kickoff_time": "date"})], ignore_index=True)
    alone = rate_opponents(team_matches, one)["opponent_defence_last3"].iloc[0]
    together = rate_opponents(team_matches, two)["opponent_defence_last3"].iloc[0]
    assert alone == pytest.approx(together), "an extra placeholder moved an existing rating"


def _per_team_mean_rating(team_matches: pd.DataFrame, window: int = 5) -> pd.Series:
    """Mean rating per club, from either source, computed the same way both times."""
    rated = team_matches.copy()
    rated[f"opponent_defence_last{window}"] = rate_team_matches(team_matches, windows=(window,))[
        f"opponent_defence_last{window}"]
    return rated.groupby("team")[f"opponent_defence_last{window}"].mean().dropna()


def test_the_serving_adapter_agrees_with_the_training_adapter_on_real_data():
    """The load-bearing test for promotion. The fitting path reads the FPL
    archive; the serving path reads football-data.co.uk via `to_team_perspective`.
    They are different files describing the same matches, so the ratings should
    agree. If they do not, promoting the feature would train on one definition of
    "defence" and serve another -- the failure mode of NFL `0628c6d`."""
    from pl_predictor.data import football_data
    from pl_predictor.data.team_names import to_canonical

    season = "2024-25"
    history = load_player_gw_history(seasons=[season])
    matches_df = football_data.load_training_data(seasons=[season])

    # Training side: the archive, via `build_team_match_history`.
    trained = _per_team_mean_rating(_generic_team_matches(history))
    trained.index = [to_canonical(str(name), "fpl_api") for name in trained.index]

    # Serving side: football-data.co.uk, via `to_team_perspective`.
    served = _per_team_mean_rating(team_matches_from_matches_df(matches_df))

    comparison = pd.DataFrame({"serving": served, "training": trained}).dropna()

    # Only 18 of 20 clubs land here, and the two that drop out are Man Utd and
    # Tottenham -- both `to_canonical` misses on the FPL side. That is the whole
    # argument for building the fitting path out of the archive alone: a name
    # mapping that loses two clubs quietly costs the feature for every player who
    # faces them. Here the loss is visible; in the model it would not be.
    assert len(comparison) == 18, (
        f"expected 18 comparable clubs, got {len(comparison)}; the two name spaces "
        f"have drifted apart: {sorted(set(trained.index) ^ set(served.index))}")

    # The two sources agree exactly on 2024-25 -- measured max per-club
    # difference 0.0000 goals/match, correlation 1.0000. The tolerances are set
    # just above what a couple of *corrected* scorelines would move them (one
    # correction shifts a club's 38-match mean by ~0.026) and far below what a
    # definitional difference produces (~0.3+), so this test fails on a drifted
    # definition rather than merely warning about it.
    assert comparison["serving"].corr(comparison["training"]) > 0.99, (
        f"the two adapters disagree about which defences are leaky:\n{comparison.round(3)}")
    worst = (comparison["serving"] - comparison["training"]).abs().max()
    assert worst < 0.15, (
        f"worst per-club disagreement is {worst:.3f} goals/match, too large to train on one "
        f"definition and serve another:\n"
        f"{(comparison['serving'] - comparison['training']).abs().sort_values(ascending=False).head(5).round(3)}")
