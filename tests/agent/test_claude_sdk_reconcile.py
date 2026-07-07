"""Task 6: history reconciliation state machine (pure logic, no SDK/threads/RBAC).

The two false-positive guards (normal-turn and system-prompt-change) are the
load-bearing tests: a false "reseed" on a normal turn throws away the live SDK
session; a false "no reseed" on a rewrite corrupts the conversation.
"""


def _mk_session():
    from agent.claude_sdk_client import _SdkSession
    return _SdkSession(session_id="s1")


def test_normal_turn_sends_only_new_user_message_no_reseed():
    sess = _mk_session()
    # Shadow already contains a prior exchange (system stripped).
    sess.shadow = [{"role": "user", "content": "u1"},
                   {"role": "assistant", "content": "a1"}]
    incoming = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"}]
    decision = sess.reconcile(incoming)
    assert decision.reseed is False
    assert decision.new_messages == [{"role": "user", "content": "u2"}]


def test_changed_system_prompt_does_not_reseed():
    sess = _mk_session()
    sess.shadow = [{"role": "user", "content": "u1"},
                   {"role": "assistant", "content": "a1"}]
    incoming = [{"role": "system", "content": "DIFFERENT SYSTEM PROMPT"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"}]
    decision = sess.reconcile(incoming)
    assert decision.reseed is False   # system change is applied via option, not a reseed


def test_compression_rewrite_triggers_reseed():
    sess = _mk_session()
    sess.shadow = [{"role": "user", "content": "u1"},
                   {"role": "assistant", "content": "a1"},
                   {"role": "user", "content": "u2"}]
    # History was rewritten (shadow is no longer a prefix of incoming non-system).
    incoming = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "[compressed summary]"},
                {"role": "user", "content": "u3"}]
    decision = sess.reconcile(incoming)
    assert decision.reseed is True
    assert decision.reseed_reason  # non-empty


def test_missing_prefix_after_restart_triggers_reseed():
    sess = _mk_session()
    sess.shadow = []  # fresh process, nothing known yet, but incoming has history
    incoming = [{"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"}]
    decision = sess.reconcile(incoming)
    assert decision.reseed is True


def test_fresh_first_turn_single_user_message_no_reseed():
    sess = _mk_session()
    sess.shadow = []
    incoming = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "hello"}]
    decision = sess.reconcile(incoming)
    assert decision.reseed is False
    assert decision.new_messages == [{"role": "user", "content": "hello"}]


def test_render_history_for_reseed_nonempty():
    from agent.claude_sdk_client import _render_history_for_reseed
    msgs = [{"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"}]
    out = _render_history_for_reseed(msgs, "claude-opus-4-8", None)
    assert isinstance(out, str) and out.strip()
