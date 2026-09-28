"""network_blocked.py — the test network guard's refusal, as a type.

`tests/conftest.py::_block_network` patches `socket.socket.connect` and
`socket.create_connection` during the offline suite so a test that needs
uncached data fails fast and says how to warm it. But every
`_fetch_with_retry` wrapper in this package retries broad `Exception` with
2s+4s sleeps -- and a `RuntimeError` is an `Exception`, so a guard refusal
was retried as though it were a transient upstream flake (~6s of sleeps
per blocked fetch) and then surfaced as `Failed after N attempts`,
misattributing a policy refusal to an outage.

A blocked network is a *known* state, not a transient failure: retrying it
is never correct. This type makes the refusal recognizable without
message-sniffing (rewording the guard's hint must never silently
reintroduce the sleeps), and it subclasses `RuntimeError` so every
existing degrade-on-failure path (`except RuntimeError` in `load_xg_data`,
`_load_season_*`, `fetch_current_season_partial`, ...) catches it exactly
as before.

Production impact: none, by construction. Nothing in production raises
this -- only the test guard does -- so the `except NetworkBlockedError:
raise` clauses in the retry wrappers never trigger outside the suite.
"""

from __future__ import annotations


class NetworkBlockedError(RuntimeError):
    """The suite's network guard refused an outbound connection.

    Raised only by `tests/conftest.py::_block_network`. Terminal by
    contract: retry wrappers must re-raise it immediately, with no sleeps.
    The message carries the why (offline by policy) and the remedy
    (`PL_ALLOW_NETWORK=1` warming / `@pytest.mark.network` opt-in).
    """
