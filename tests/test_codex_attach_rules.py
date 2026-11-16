import importlib.util
import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


HOOK_PATH = Path(__file__).resolve().parents[1] / ".codex" / "hooks" / "attach_rules.py"
SPEC = importlib.util.spec_from_file_location("codex_attach_rules", HOOK_PATH)
HOOK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOOK)


class CodexAttachRulesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_state_dir = HOOK.STATE_DIR
        HOOK.STATE_DIR = Path(self.temp_dir.name) / "state"

    def tearDown(self):
        HOOK.STATE_DIR = self.original_state_dir
        self.temp_dir.cleanup()

    def test_cursor_rules_parse_with_expected_scopes(self):
        rules = {rule.path.stem: rule for rule in HOOK.load_rules()}

        self.assertEqual(set(rules), {"workflow", "python-application", "testing", "deployment"})
        self.assertTrue(rules["workflow"].always)
        self.assertFalse(rules["testing"].always)
        self.assertEqual(
            rules["deployment"].globs,
            [
                "gigaam/Dockerfile*",
                "docker-compose*.yml",
                ".env.dist",
                ".github/workflows/*.yml",
            ],
        )

    def test_cursor_and_claude_rule_bodies_and_scopes_match(self):
        rules = {rule.path.stem: rule for rule in HOOK.load_rules()}

        for name, cursor_rule in rules.items():
            with self.subTest(rule=name):
                text = (HOOK.REPO_ROOT / ".claude" / "rules" / f"{name}.md").read_text()
                if cursor_rule.always:
                    self.assertFalse(cursor_rule.globs)
                    self.assertFalse(text.startswith("---"))
                    claude_body = text.strip()
                else:
                    end = text.find("\n---", 3)
                    self.assertGreaterEqual(end, 0)
                    frontmatter = text[3:end]
                    claude_paths = re.findall(r'^\s*-\s+"([^"]+)"\s*$', frontmatter, re.M)
                    self.assertEqual(claude_paths, cursor_rule.globs)
                    claude_body = text[end + 4 :].strip()

                self.assertEqual(claude_body, cursor_rule.body)

    def test_globs_match_project_files_without_crossing_path_segments(self):
        python_glob = HOOK.glob_to_regex("gigaam/*.py")
        tests_glob = HOOK.glob_to_regex("tests/**/*.py")

        self.assertTrue(python_glob.match("gigaam/asr.py"))
        self.assertFalse(python_glob.match("gigaam/nested/asr.py"))
        self.assertTrue(tests_glob.match("tests/test_uploads.py"))
        self.assertTrue(tests_glob.match("tests/api/test_uploads.py"))

    def test_patched_paths_reads_edit_payloads_and_rejects_paths_outside_repo(self):
        absolute_path = (HOOK.REPO_ROOT / "gigaam" / "asr.py").as_posix()
        payload = {
            "patch": (
                "*** Add File: tests/new_test.py\n"
                "*** Update File: gigaam/main.py\n"
                "*** Delete File: docker-compose.dist.yml\n"
                "*** Move to: tests/renamed_test.py\n"
            ),
            "file_path": absolute_path,
            "destination": "../outside.py",
        }

        self.assertEqual(
            HOOK.patched_paths(payload),
            [
                "gigaam/asr.py",
                "tests/new_test.py",
                "gigaam/main.py",
                "docker-compose.dist.yml",
                "tests/renamed_test.py",
            ],
        )

    def test_patch_markers_in_content_are_not_treated_as_edit_paths(self):
        content = "Example text\n*** Add File: tests/not_being_edited.py\n"

        self.assertEqual(HOOK.patched_paths({"content": content}), [])
        self.assertEqual(
            HOOK.patched_paths("*** Add File: tests/new_test.py\n"),
            ["tests/new_test.py"],
        )

    def test_session_start_delivers_only_always_on_rules(self):
        rules = HOOK.load_rules()
        delivered = []

        did_deliver = HOOK.deliver_session_start(
            "session-start-test", "startup", [rule for rule in rules if rule.always], delivered.append
        )

        self.assertTrue(did_deliver)
        self.assertEqual([rule.path.stem for rule in delivered[0]], ["workflow"])

    def test_pre_tool_use_dispatches_matching_scoped_rules(self):
        cases = [
            ("gigaam/main.py", "Python application boundaries", "# Tests"),
            ("tests/test_uploads.py", "# Tests", "Deployment and image builds"),
            (".env.dist", "Deployment and image builds", "Python application boundaries"),
        ]

        for index, (path, expected, unrelated) in enumerate(cases):
            with self.subTest(path=path):
                payload = {
                    "hook_event_name": "PreToolUse",
                    "session_id": f"pre-tool-{index}",
                    "tool_input": {"file_path": path},
                }
                output = io.StringIO()
                with (
                    patch.object(HOOK.sys, "stdin", io.StringIO(json.dumps(payload))),
                    redirect_stdout(output),
                ):
                    self.assertEqual(HOOK.main(), 0)

                context = json.loads(output.getvalue())["hookSpecificOutput"]["additionalContext"]
                self.assertIn(expected, context)
                self.assertNotIn(unrelated, context)

    def test_pre_tool_use_without_valid_session_id_delivers_without_shared_state(self):
        for session_id in (None, 42):
            with self.subTest(session_id=session_id):
                payload = {
                    "hook_event_name": "PreToolUse",
                    "tool_input": {"file_path": "tests/test_uploads.py"},
                }
                if session_id is not None:
                    payload["session_id"] = session_id
                output = io.StringIO()
                with (
                    patch.object(HOOK.sys, "stdin", io.StringIO(json.dumps(payload))),
                    redirect_stdout(output),
                ):
                    self.assertEqual(HOOK.main(), 0)

                context = json.loads(output.getvalue())["hookSpecificOutput"]["additionalContext"]
                self.assertIn("# Tests", context)

        self.assertFalse(HOOK.STATE_DIR.exists())

    def test_session_start_rule_load_failure_does_not_clear_unknown_session_state(self):
        sent = {".cursor/rules/testing.mdc"}
        self.assertTrue(HOOK.save_sent("unknown", sent))
        payload = {"hook_event_name": "SessionStart", "source": "startup"}
        output = io.StringIO()

        with (
            patch.object(HOOK, "load_rules", side_effect=RuntimeError("invalid rule file")),
            patch.object(HOOK.sys, "stdin", io.StringIO(json.dumps(payload))),
            redirect_stdout(output),
        ):
            self.assertEqual(HOOK.main(), 0)

        self.assertEqual(HOOK.load_sent("unknown"), sent)

    def test_scoped_rule_is_delivered_once_per_session(self):
        rule = next(rule for rule in HOOK.load_rules() if rule.path.stem == "testing")
        first_delivery = []
        second_delivery = []

        self.assertTrue(HOOK.deliver_rule_context("scoped-test", [rule], first_delivery.append))
        self.assertFalse(HOOK.deliver_rule_context("scoped-test", [rule], second_delivery.append))
        self.assertEqual(first_delivery, [[rule]])
        self.assertEqual(second_delivery, [])

    def test_state_directory_is_created_with_private_permissions(self):
        HOOK.ensure_state_dir()

        if os.name != "nt":
            self.assertEqual(HOOK.STATE_DIR.stat().st_mode & 0o777, 0o700)

    def test_state_directory_is_namespaced_by_repository_and_user(self):
        repo_path = Path("/workspace/repo-a")
        user_namespace = "user-a"

        base = HOOK.state_dir_for(repo_path, user_namespace)
        self.assertNotEqual(HOOK.state_dir_for(Path("/workspace/repo-b"), user_namespace), base)
        self.assertNotEqual(HOOK.state_dir_for(repo_path, "user-b"), base)

    def test_state_files_do_not_collide_after_sanitizing_or_truncating_ids(self):
        self.assertNotEqual(HOOK.state_file("a/b"), HOOK.state_file("a_b"))
        self.assertNotEqual(HOOK.state_file("x" * 130 + "a"), HOOK.state_file("x" * 130 + "b"))

    def test_non_object_json_payload_fails_open(self):
        for payload in ([], None, False, 42, "text"):
            with self.subTest(payload=payload):
                output = io.StringIO()
                with (
                    patch.object(HOOK.sys, "stdin", io.StringIO(json.dumps(payload))),
                    redirect_stdout(output),
                ):
                    self.assertEqual(HOOK.main(), 0)

                self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
