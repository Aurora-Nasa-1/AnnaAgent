import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from typer.testing import CliRunner

from anna_agent.assets import manifest_path
from anna_agent.cli import app
from anna_agent.runtime import load_state

runner = CliRunner()


def _fake_full_state(case_file: Path, progress_callback=None):
    if progress_callback:
        progress_callback("fake", "Building full prompt state")
    return {
        "schema_version": 1,
        "mode": "full",
        "case_id": case_file.stem,
        "seeker_id": "seeker-1",
        "portrait": {},
        "report": {},
        "previous_conversations": [],
        "prompt": "Act as a seeker.",
        "complaint_chain": [],
        "configuration": {},
        "metadata": {"source_file": str(case_file)},
    }


def test_workspace_to_batch_full_prompt_journey(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"

    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output
    assert "Research & Citation" in result.output
    assert "https://aclanthology.org/2025.findings-acl.1192/" in result.output
    assert "https://github.com/sci-m-wang/AnnaAgent" in result.output
    assert "please star" in result.output
    assert "please cite" in result.output
    assert (workspace / "settings.yaml").exists()
    assert (workspace / "assets" / "anna-assets.json").exists()
    case_file = workspace / "cases" / "family_stress_case.json"
    assert case_file.exists()

    result = runner.invoke(app, ["assets", "list", "--workspace", str(workspace)])
    assert result.exit_code == 0, result.output
    assert "emotion-sft" in result.output

    result = runner.invoke(app, ["data", "validate", str(case_file)])
    assert result.exit_code == 0, result.output

    state_file = workspace / "prompts" / "case.state.json"
    monkeypatch.setattr("anna_agent.cli.build_full_state", _fake_full_state)
    result = runner.invoke(
        app,
        [
            "init",
            "full",
            str(case_file),
            "--workspace",
            str(workspace),
            "--out",
            str(state_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Initialization · full" in result.output
    assert "Running:" in result.output
    assert "Research & Citation" in result.output
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["mode"] == "full"
    assert state["prompt"]

    result = runner.invoke(app, ["init", "from-prompt", str(state_file)])
    assert result.exit_code == 0, result.output
    assert "prompt_chars" in result.output

    out_dir = workspace / "runs" / "batch"
    result = runner.invoke(
        app,
        [
            "run",
            "batch",
            "--workspace",
            str(workspace),
            "--case",
            "cases/*.json",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "summary.jsonl").exists()


def test_assets_download_reports_unconfigured_assets(tmp_path: Path):
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output
    manifest = json.loads(manifest_path(workspace).read_text(encoding="utf-8"))
    manifest["presets"] = {"local-test": ["missing-url"]}
    manifest["assets"] = [
        {
            "name": "missing-url",
            "kind": "dataset",
            "target": "assets/missing-url",
            "source": {"type": "url", "url": ""},
        }
    ]
    manifest_path(workspace).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        ["assets", "download", "local-test", "--workspace", str(workspace)],
    )

    assert result.exit_code == 0, result.output
    assert "unconfigured" in result.output


def test_assets_list_resolves_manifest_absolute_targets(tmp_path: Path):
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output
    manifest = json.loads(manifest_path(workspace).read_text(encoding="utf-8"))
    absolute_target = Path("/tmp/a")
    manifest["assets"] = [
        {
            "name": "emotion-sft",
            "kind": "model",
            "target": str(absolute_target),
            "source": {
                "type": "huggingface",
                "repo_id": "sci-m-wang/Emotion_inferencer-Qwen2.5-7B-Instruct",
                "repo_type": "model",
                "revision": "main",
            },
        }
    ]
    manifest_path(workspace).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = runner.invoke(app, ["assets", "list", "--workspace", str(workspace)])

    assert result.exit_code == 0, result.output
    assert str(absolute_target) in result.output


def test_assets_download_target_override_requires_single_asset(tmp_path: Path):
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "assets",
            "download",
            "paper",
            "--workspace",
            str(workspace),
            "--target",
            str(tmp_path / "one-target"),
        ],
    )

    assert result.exit_code != 0


def test_config_secrets_writes_hidden_values_to_dotenv(tmp_path: Path):
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        ["config", "secrets", "--workspace", str(workspace)],
        input="base-secret\nembed-secret\n",
    )

    assert result.exit_code == 0, result.output
    dotenv_text = (workspace / ".env").read_text(encoding="utf-8")
    assert "ANNA_ENGINE_API_KEY=base-secret" in dotenv_text
    assert "ANNA_ENGINE_EMBEDDING_API_KEY=embed-secret" in dotenv_text
    assert "base-secret" not in result.output


def test_chat_state_uses_stage_based_rich_ui(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output
    state_file = workspace / "prompts" / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "full",
                "case_id": "case-1",
                "seeker_id": "seeker-1",
                "portrait": {},
                "report": {},
                "previous_conversations": [],
                "prompt": "Act as a seeker.",
            }
        ),
        encoding="utf-8",
    )

    class FakeFrozenPromptSession:
        def __init__(self, state):
            self.state = state
            self.last_turn_context = {
                "emotion": "sadness",
                "complaint_stage": 1,
                "complaint": "family pressure",
                "memory_used": False,
            }

        def chat(self, message: str) -> str:
            assert message == "你好"
            return "我最近有点累。"

    monkeypatch.setattr("anna_agent.cli.FrozenPromptSession", FakeFrozenPromptSession)

    result = runner.invoke(
        app,
        ["chat", "--workspace", str(workspace), "--state", str(state_file)],
        input="你好\nq\n",
    )

    assert result.exit_code == 0, result.output
    assert "AnnaAgent Chat" in result.output
    assert "Stage 1/2" in result.output
    assert "Stage 2/2" in result.output
    assert "Counselor" in result.output
    assert "Seeker" in result.output
    assert "我最近有点累" in result.output
    assert "ChatCompletion" not in result.output


def test_chat_compact_ui_shows_initialization_progress(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output

    class FakeMsPatient:
        def __init__(self, portrait, report, conversations, progress_callback=None):
            self.last_turn_context = {}
            if progress_callback:
                progress_callback("fake", "Preparing compact initialization")

        def chat(self, message: str) -> str:
            return "我最近有点累。"

    monkeypatch.setattr("anna_agent.ms_patient.MsPatient", FakeMsPatient)

    result = runner.invoke(
        app,
        ["chat", "--workspace", str(workspace)],
        input="q\n",
    )

    assert result.exit_code == 0, result.output
    assert "正在运行：Preparing compact initialization" in result.output


def test_chat_debug_ui_shows_internal_state(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output
    state_file = workspace / "prompts" / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "full",
                "case_id": "case-1",
                "seeker_id": "seeker-1",
                "portrait": {},
                "report": {},
                "previous_conversations": [],
                "prompt": "Act as a seeker.",
            }
        ),
        encoding="utf-8",
    )

    class FakeFrozenPromptSession:
        def __init__(self, state):
            self.state = state
            self.last_turn_context = {
                "emotion": "sadness",
                "complaint_stage": 1,
                "complaint": "family pressure",
                "memory_used": False,
            }

        def chat(self, message: str) -> str:
            return "我最近有点累。"

    monkeypatch.setattr("anna_agent.cli.FrozenPromptSession", FakeFrozenPromptSession)

    result = runner.invoke(
        app,
        [
            "chat",
            "--workspace",
            str(workspace),
            "--state",
            str(state_file),
            "--debug-ui",
        ],
        input="你好\nq\n",
    )

    assert result.exit_code == 0, result.output
    assert "调试双模式" in result.output
    assert "本轮内部状态" in result.output
    assert "sadness" in result.output


def test_prompt_only_state_is_rejected(tmp_path: Path):
    state_file = tmp_path / "prompt-only.json"
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "prompt_only",
                "portrait": {},
                "report": {},
                "previous_conversations": [],
                "prompt": "Wrong shortcut prompt.",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt_only states are not supported"):
        load_state(state_file)


def test_initialize_full_connection_error_is_concise(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output
    case_file = workspace / "cases" / "family_stress_case.json"

    class APIConnectionError(Exception):
        pass

    def fake_build_full_state(case_file, progress_callback=None):
        raise APIConnectionError("Connection error.")

    monkeypatch.setattr("anna_agent.cli.build_full_state", fake_build_full_state)

    result = runner.invoke(
        app,
        [
            "init",
            "full",
            str(case_file),
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 1
    assert "Model service connection failed" in result.output
    assert "Run anna doctor" in result.output
    assert "Traceback" not in result.output


def test_model_auth_error_is_concise_and_redacted(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    result = runner.invoke(app, ["create", str(workspace)])
    assert result.exit_code == 0, result.output
    monkeypatch.setenv("ANNA_ENGINE_API_KEY", "secret-123")

    class AuthenticationError(Exception):
        pass

    class FakeCompletions:
        def create(self, **kwargs):
            raise AuthenticationError("401 Invalid API Key secret-123")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(
        "anna_agent.cli.backbone.get_openai_client", lambda: FakeClient()
    )

    result = runner.invoke(
        app,
        [
            "test",
            "model",
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 1
    assert "Model service authentication failed" in result.output
    assert "ANNA_ENGINE_API_KEY" in result.output
    assert "secret-123" not in result.output
    assert "Traceback" not in result.output
