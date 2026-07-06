"""Cron-test fixtures.

Provides a default ``HERMES_MODEL`` for cron run_job tests so each one
doesn't have to spell out a model. The global conftest blanks
HERMES_MODEL hermetically; without this autouse fixture every cron test
that exercises ``run_job`` would hit the fail-fast guard added in
``cron/scheduler.py`` (see issue #23979) and have to be rewritten.

Tests that specifically need ``HERMES_MODEL`` unset — model-resolution
edge cases — call ``monkeypatch.delenv("HERMES_MODEL", raising=False)``
inside the test, which overrides this fixture's value for that scope.
"""

import pytest


@pytest.fixture(autouse=True)
def _default_cron_test_model(monkeypatch):
    """Pin a default HERMES_MODEL so cron run_job tests have a resolvable model."""
    monkeypatch.setenv("HERMES_MODEL", "test-cron-default-model")
    yield


@pytest.fixture(autouse=True)
def _approve_all_cron_scripts_by_default(request, monkeypatch):
    """Existing cron tests exercise script EXECUTION mechanics, not the approval
    gate (Task 7). Default them to 'approved' so the run-time pin inside
    ``_run_job_script`` doesn't block them. Tests that specifically verify the
    pin opt OUT via the ``real_script_pin`` marker (see
    ``tests/cron/test_script_pin.py``).

    The gate imports ``is_approved`` function-locally
    (``from cron.script_registry import is_approved``), so the binding is looked
    up at call time — patching ``cron.script_registry.is_approved`` here is the
    seam the scheduler sees.

    RULE FOR FUTURE AUTHORS: any NEW test that intends to exercise the run-time
    approval gate (assert an unapproved script is blocked, or that an approved
    one runs) MUST carry ``@pytest.mark.real_script_pin`` — otherwise this
    default bypass silently stubs the gate to True and the test proves nothing
    about approval.
    """
    if request.node.get_closest_marker("real_script_pin"):
        return
    import cron.script_registry as sr
    monkeypatch.setattr(sr, "is_approved", lambda *a, **k: True)
