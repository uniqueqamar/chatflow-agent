"""Tests for BaseChannel and CLIChannel."""

import pytest

from chatflow_agent.channels.base import BaseChannel, ChannelError
from chatflow_agent.channels.cli import CLIChannel
from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.runner import Runner


class DummyChannel(BaseChannel):
    """Minimal concrete channel for base tests."""

    def run(self, **kwargs) -> None:
        pass


class MockEngine(GeminiEngine):
    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply = reply

    async def generate_turn_async(self, agent, history):
        return EngineTurnResult(text=self.reply)


def test_base_channel_detached_error() -> None:
    channel = DummyChannel()
    with pytest.raises(ChannelError) as exc_info:
        channel.dispatch(session_id="s1", message="hello")
    assert "No Runner is attached" in str(exc_info.value)


@pytest.mark.asyncio
async def test_base_channel_detached_error_async() -> None:
    channel = DummyChannel()
    with pytest.raises(ChannelError) as exc_info:
        await channel.dispatch_async(session_id="s1", message="hello")
    assert "No Runner is attached" in str(exc_info.value)


def test_base_channel_attach_and_dispatch() -> None:
    agent = Agent(name="Greeter", instructions="Greet.")
    runner = Runner(starting_agent=agent, engine=MockEngine("Welcome!"))

    channel = DummyChannel()
    ret = channel.attach(runner)
    assert ret is channel  # Fluent interface

    resp = channel.dispatch(session_id="test_session", message="Hi")
    assert resp.content == "Welcome!"
    assert resp.active_agent_name == "Greeter"


def test_cli_channel_send_message() -> None:
    agent = Agent(name="CLI Assistant", instructions="Assist user via CLI.")
    runner = Runner(starting_agent=agent, engine=MockEngine("Command processed."))

    cli = CLIChannel(session_id="custom_cli_user")
    cli.attach(runner)

    resp = cli.send_message("Run diagnostic")
    assert resp.content == "Command processed."
    assert resp.active_agent_name == "CLI Assistant"


def make_cli_for_loop_tests() -> CLIChannel:
    """Attach a lightweight runner without invoking an external model."""
    agent = Agent(name="CLI Assistant", instructions="Help.")
    cli = CLIChannel(session_id="loop_test")
    cli.attach(Runner(starting_agent=agent, engine=MockEngine("Done.")))
    return cli


@pytest.mark.parametrize("exit_command", ["exit", "quit", "salir", "EXIT"])
def test_cli_run_exit_commands(monkeypatch, capsys, exit_command: str) -> None:
    cli = make_cli_for_loop_tests()
    prompts = iter([exit_command])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))
    cli.run()
    assert "Goodbye!" in capsys.readouterr().out


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_cli_run_interruptions(monkeypatch, capsys, exception: type[Exception]) -> None:
    cli = make_cli_for_loop_tests()

    def interrupt(_prompt: str) -> str:
        raise exception()

    monkeypatch.setattr("builtins.input", interrupt)
    cli.run()
    assert "Session ended by user." in capsys.readouterr().out


def test_cli_run_ignores_empty_input(monkeypatch, capsys) -> None:
    from chatflow_agent.types import AgentResponse

    cli = make_cli_for_loop_tests()
    prompts = iter(["", "   ", "Hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))
    calls = []

    def dispatch(session_id: str, message: str) -> AgentResponse:
        calls.append((session_id, message))
        return AgentResponse(content="Done.", active_agent_name="CLI Assistant")

    monkeypatch.setattr(cli, "dispatch", dispatch)
    cli.run()
    assert calls == [("loop_test", "Hello")]
    assert "[CLI Assistant]: Done." in capsys.readouterr().out


def test_cli_run_prints_handoff(monkeypatch, capsys) -> None:
    from chatflow_agent.types import AgentResponse, Handoff

    cli = make_cli_for_loop_tests()
    prompts = iter(["transfer", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))
    monkeypatch.setattr(
        cli,
        "dispatch",
        lambda _session, _message: AgentResponse(
            content="Transferred",
            active_agent_name="Specialist",
            handoff=Handoff(target_agent_name="Specialist", reason="Specialized task"),
        ),
    )
    cli.run()
    output = capsys.readouterr().out
    assert "[Transferred to Specialist]" in output
    assert "Reason: Specialized task" in output


def test_cli_run_recovers_after_dispatch_error(monkeypatch, capsys) -> None:
    cli = make_cli_for_loop_tests()
    prompts = iter(["first", "second", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))
    calls = []

    def dispatch(_session_id: str, message: str) -> None:
        calls.append(message)
        raise RuntimeError("temporary failure")

    monkeypatch.setattr(cli, "dispatch", dispatch)
    cli.run()
    captured = capsys.readouterr()
    assert calls == ["first", "second"]
    assert captured.err.count("Error during processing: temporary failure") == 2
    assert "Goodbye!" in captured.out
