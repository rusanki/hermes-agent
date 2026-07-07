"""Ensure the verified Slack ``user_id`` is forwarded from every
``get_pre_tool_call_block_message`` call site.

Authorization in the ``pre_tool_call`` policy hook rides on the verified
session ``user_id`` (a server-set contextvar, not agent-spoofable). There are
three invocations of ``get_pre_tool_call_block_message``:

* ``agent/tool_executor.py`` — already forwards ``user_id``.
* ``agent/agent_runtime_helpers.py`` — fixed here.
* ``model_tools.py`` — fixed here.

If a call site omits ``user_id``, the hook receives ``""`` and resolves to the
``system``/``member`` role, silently bypassing any admin-only gate. These
source-level assertions guard against that regression at each edited site.
"""

import inspect

import agent.agent_runtime_helpers as arh
import model_tools as mt


def _balanced_call(src: str, i: int) -> str:
    """Return the full (possibly multi-line) ``(...)`` call text starting at
    ``src[i]`` (the start of the call name), balancing parentheses."""
    j = src.index("(", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "(":
            depth += 1
        elif src[k] == ")":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    return src[i:]


def _hook_call_window(module) -> str:
    """Return the full text of the real ``get_pre_tool_call_block_message(...)``
    call in ``module``'s source.

    Both modules also mention the name in comments/imports, so match the actual
    invocation by requiring the balanced window to contain ``middleware_trace=``
    (a keyword present on every real call site)."""
    src = inspect.getsource(module)
    needle = "get_pre_tool_call_block_message("
    start = 0
    while True:
        i = src.index(needle, start)
        window = _balanced_call(src, i)
        if "middleware_trace=" in window:
            return window
        start = i + len(needle)


def test_agent_runtime_helpers_forwards_user_id():
    assert "user_id=" in _hook_call_window(arh)


def test_model_tools_forwards_user_id():
    assert "user_id=" in _hook_call_window(mt)


def test_helpers_return_verified_session_user_id():
    """Behavioral guard: each edited site's ``_get_user_id_for_hooks`` resolves the
    verified session ``user_id`` from the contextvar — not a hardcoded ``""``.

    A textual ``user_id=`` assertion cannot catch a regression where the kwarg is
    present but wired to the wrong value (e.g. ``user_id=""``). This drives the
    real accessor: set the session contextvar, then confirm each helper returns it.
    """
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform="slack", user_id="U_VERIFIED")
    try:
        assert arh._get_user_id_for_hooks() == "U_VERIFIED"
        assert mt._get_user_id_for_hooks() == "U_VERIFIED"
    finally:
        clear_session_vars(tokens)

    # Outside any session, both degrade to "" (the pre-fix, fail-safe behavior).
    assert arh._get_user_id_for_hooks() == ""
    assert mt._get_user_id_for_hooks() == ""
