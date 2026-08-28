from __future__ import annotations

import unittest
from pathlib import Path

from scripts.submit_locany_cpt import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    load_resource,
    parse_args,
    render_job,
)


class SubmitLocateAnythingCPTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_text = (
            PROJECT_ROOT / "locany_cpt_v4_a100x4_smoke_merlin.yaml"
        ).read_text(encoding="utf-8")

    def test_yg_resource_keeps_default_group_and_no_explicit_queue(self) -> None:
        resource = load_resource(DEFAULT_CONFIG, "yg")
        rendered = render_job(
            self.base_text,
            cluster="yg",
            resource=resource,
        )
        self.assertIn("groupIds:\n        - 1602", rendered)
        self.assertNotIn("queueName:", rendered)
        self.assertIn('INSTALL_SYSTEM_RUNTIME_DEPS: "1"', rendered)

    def test_aiai_locate_renders_group_queue_and_identifiable_job(self) -> None:
        resource = load_resource(DEFAULT_CONFIG, "aiai_locate")
        rendered = render_job(
            self.base_text,
            cluster="aiai_locate",
            resource=resource,
        )
        self.assertIn("groupIds:\n        - 2146", rendered)
        self.assertNotIn("groupIds:\n        - 1602", rendered)
        self.assertIn(
            "queueName: compute-3302-yg-cloudnative-ai-aiai.locate-guarantee",
            rendered,
        )
        self.assertIn("name: 'locany-cpt-v4-v2-a100x4-smoke-aiai-locate'", rendered)
        self.assertIn("[aiai_locate]", rendered)

    def test_runtime_dependency_install_can_be_disabled_explicitly(self) -> None:
        rendered = render_job(
            self.base_text,
            cluster="yg",
            resource=load_resource(DEFAULT_CONFIG, "yg"),
            install_system_runtime_deps=False,
        )
        self.assertIn('INSTALL_SYSTEM_RUNTIME_DEPS: "0"', rendered)
        self.assertNotIn('INSTALL_SYSTEM_RUNTIME_DEPS: "1"', rendered)

    def test_cli_cluster_selector(self) -> None:
        args = parse_args(["--mode", "smoke", "--cluster", "aiai_locate"])
        self.assertEqual(args.mode, "smoke")
        self.assertEqual(args.cluster, "aiai_locate")
        self.assertTrue(args.install_system_runtime_deps)

    def test_unknown_cluster_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown --cluster"):
            load_resource(DEFAULT_CONFIG, "not_a_resource_group")


if __name__ == "__main__":
    unittest.main()
