"""Guard tests: the network block is real, restores, and does not leak.

A guard that cannot fail, or that leaks, is worse than no guard -- it is
trusted. These are pinned against deliberately broken copies, the same
convention test_deploy_workflow.py uses.
"""
from __future__ import annotations

import socket
import types

import pytest


def test_the_network_is_blocked_during_a_test():
    """The suite must not be able to reach a third party while it runs.

    Checked from inside a running test, because a fixture that is merely
    installed in conftest is not proof it fires.
    """
    with pytest.raises(RuntimeError, match="outbound network blocked"):
        socket.create_connection(("example.com", 80), timeout=1)


def test_sockets_are_restored_after_a_test():
    """The block must not outlive the test that asked for it.

    The autouse fixture wraps THIS test too, so asserting "the socket is
    restored" here would be comparing the fixture's own patch to itself --
    vacuous, or failing for the wrong reason depending on how it is written.
    So this drives the fixture's generator directly: install, check it blocks,
    tear down, then check the socket is back to the object captured at import.
    """
    from conftest import (
        PRISTINE_CREATE_CONNECTION,
        PRISTINE_SOCKET_CONNECT,
        _block_network,
    )

    # Unwrap the pytest fixture to get the raw generator, and hand it a request
    # with no `network` marker. `__wrapped__` is what pytest itself uses.
    raw = _block_network.__wrapped__
    request = type(
        "R", (), {"node": type("N", (), {"get_closest_marker": staticmethod(lambda name: None)})()}
    )()
    # This test is itself inside the autouse guard, so `socket` is already
    # patched on entry. The invariant is therefore "teardown puts back exactly
    # what was there when this generator started", not "puts back the stdlib" --
    # the stdlib originals are asserted separately below, from the conftest
    # module's import-time captures, which is the only place they can be seen
    # from inside a guarded test.
    before_create = socket.create_connection
    before_connect = socket.socket.connect

    gen = raw(request)
    next(gen)
    try:
        assert socket.create_connection is not before_create, (
            "the guard did not install its block at all -- this test would pass vacuously"
        )
        with pytest.raises(RuntimeError, match="outbound network blocked"):
            socket.create_connection(("example.com", 80), timeout=1)
    finally:
        with pytest.raises(StopIteration):
            next(gen)

    assert socket.create_connection is before_create, (
        "the teardown did not restore socket.create_connection to its entry state"
    )
    assert socket.socket.connect is before_connect, (
        "the teardown did not restore socket.socket.connect to its entry state"
    )
    # The entries themselves are the stdlib originals on an unwarmed session,
    # which is what makes the outer guard's own teardown trustworthy.
    # `socket.socket.connect` is a C method_descriptor and has no __module__, so
    # it is identified by being the builtin rather than a Python function --
    # which is exactly the distinction a leaked patch would break.
    assert PRISTINE_CREATE_CONNECTION.__module__ == "socket", (
        "conftest captured a non-stdlib create_connection as pristine"
    )
    assert isinstance(PRISTINE_SOCKET_CONNECT, types.MethodDescriptorType), (
        f"conftest captured a Python-level socket.connect as pristine: {PRISTINE_SOCKET_CONNECT!r}"
    )


@pytest.mark.network
def test_the_network_marker_opts_a_test_back_in():
    """The opt-in has to actually work, or it is not an opt-in.

    This does not touch the network: it asserts the marker was seen, which is
    the only thing the fixture keys off. A suite where the escape hatch is
    broken gets a hard-blocked test with no route out.
    """
    assert socket.create_connection.__module__ == "socket"
    assert pytest.mark.network.name == "network"


def test_a_blocked_test_gets_an_actionable_message():
    """The failure must say how to get the data, not just that it was denied.

    This is the difference between a five-second fix and an afternoon. A cold
    checkout is the normal state for anyone who has not run the suite before, so
    this error is the first thing a new contributor sees from this suite.
    """
    with pytest.raises(RuntimeError) as excinfo:
        socket.create_connection(("example.com", 80), timeout=1)
    message = str(excinfo.value)
    assert "PL_ALLOW_NETWORK=1" in message, (
        f"the block does not say how to warm the cache: {message!r}"
    )
    assert "pytest.mark.network" in message, (
        f"the block does not mention the per-test opt-in: {message!r}"
    )


def test_the_guard_can_be_disabled_whole_suite(monkeypatch):
    """`PL_ALLOW_NETWORK=1` has to bypass it, or warming the cache is impossible.

    The escape hatch the guard's own failure message tells people to use. If it
    stopped working, that message would be a dead end and the only way to
    populate `data/cache/` would be to delete the guard.
    """
    from conftest import _block_network

    raw = _block_network.__wrapped__
    request = type(
        "R", (), {"node": type("N", (), {"get_closest_marker": staticmethod(lambda name: None)})()}
    )()

    # This test is itself inside the autouse guard, so the socket is already
    # patched. "Bypassed" therefore means the generator left it exactly as it
    # found it -- which is the thing a regression would break, and which is
    # checkable from here. Asserting the stdlib original is in place would be
    # asserting something the outer guard makes false regardless.
    before = socket.create_connection

    with monkeypatch.context() as m:
        m.setenv("PL_ALLOW_NETWORK", "1")
        gen = raw(request)
        next(gen)
        assert socket.create_connection is before, (
            "PL_ALLOW_NETWORK=1 installed a block anyway -- the escape hatch the failure "
            "message recommends does not work"
        )
        with pytest.raises(StopIteration):
            next(gen)

    # And the marker route bypasses it too, for the same reason.
    marked_request = type(
        "R", (), {"node": type("N", (), {"get_closest_marker": staticmethod(lambda name: True)})()}
    )()
    before = socket.create_connection
    gen = raw(marked_request)
    next(gen)
    assert socket.create_connection is before, (
        "the `network` marker installed a block anyway -- the per-test opt-in does not work"
    )
    with pytest.raises(StopIteration):
        next(gen)
