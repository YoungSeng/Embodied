"""CPU regressions for unstable attributes during ready-marker publication."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ui14_verification as verification


class VerificationRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.path = self.root / "unique_images.jsonl"
        self.path.write_bytes(b'{"id":1}\n')
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.output = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.output))
        self.sleep = self.stack.enter_context(mock.patch.object(verification.time, "sleep"))

    def test_transient_attributes_retry_only_file_and_cache_stable_read(self):
        real = verification.signature(self.path); stale = [real[0], real[1], real[2] - 1]
        with verification.verification_session(self.root) as checks, \
             mock.patch.object(verification, "signature", side_effect=[real, stale, real, real]), \
             mock.patch.object(verification, "stable_sha256", wraps=verification.stable_sha256) as reads:
            actual = checks.sha256(self.path)
            self.assertEqual(actual, hashlib.sha256(self.path.read_bytes()).hexdigest())
            self.assertEqual(reads.call_count, 2)
            self.assertEqual(checks.counts["checked"], 1)
        self.assertEqual(self.sleep.call_count, 1)
        self.assertIn("before=", self.output.getvalue()); self.assertIn("after=", self.output.getvalue())
        with verification.verification_session(self.root) as checks, \
             mock.patch.object(verification, "stable_sha256", side_effect=AssertionError("stable hash reread")):
            self.assertEqual(checks.sha256(self.path), actual)

    def test_change_after_hash_never_commits_old_digest(self):
        with verification.verification_session(self.root) as checks:
            original = checks.remember; calls = []
            def remember(path, kind, value, measured_stat=None):
                calls.append(value)
                if len(calls) == 1: self.path.write_bytes(b'{"id":"changed to another record"}\n')
                return original(path, kind, value, measured_stat)
            with mock.patch.object(checks, "remember", side_effect=remember): actual = checks.sha256(self.path)
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0], actual)
            self.assertEqual(actual, hashlib.sha256(self.path.read_bytes()).hexdigest())
            self.assertEqual(checks.counts["checked"], 1)
            self.assertEqual(len(checks.journal.rows), 1)

    def test_persistent_instability_fails_closed_after_four_attempts(self):
        with verification.verification_session(self.root) as checks, \
             mock.patch.object(verification, "stable_sha256", side_effect=
                 verification.FileChangedDuringVerification(self.path, [1, 2, 3], [1, 2, 4])) as reads:
            with self.assertRaisesRegex(ValueError, "4 unstable read attempts"):
                checks.sha256(self.path)
            self.assertEqual(reads.call_count, 4)
            self.assertEqual(self.sleep.call_count, 3)
            self.assertEqual(checks.journal.rows, {})

    def test_file_replacement_during_read_is_rejected_by_descriptor_identity(self):
        original_stat = Path.stat
        other = self.root / "other.jsonl"; other.write_bytes(self.path.read_bytes())
        real = original_stat(self.path); other_stat = original_stat(other)
        # Simulate a same-sized/timestamp replacement at the post-read path stat.
        from types import SimpleNamespace
        replaced = SimpleNamespace(st_size=real.st_size, st_mtime_ns=real.st_mtime_ns,
            st_ctime_ns=real.st_ctime_ns, st_dev=other_stat.st_dev, st_ino=other_stat.st_ino)
        self.assertNotEqual((real.st_dev, real.st_ino), (replaced.st_dev, replaced.st_ino))
        with mock.patch.object(Path, "stat", side_effect=[real, replaced]):
            with self.assertRaisesRegex(verification.FileChangedDuringVerification, "opened_file_id"):
                verification.stable_sha256(self.path)

    def test_missing_file_does_not_retry_or_cache(self):
        with verification.verification_session(self.root) as checks:
            with self.assertRaises(FileNotFoundError): checks.sha256(self.root / "missing")
            self.assertEqual(checks.journal.rows, {})
            self.sleep.assert_not_called()

    def test_path_and_descriptor_ctime_can_differ_but_each_must_be_stable(self):
        from types import SimpleNamespace
        real_fstat = verification.os.fstat
        def descriptor(fd):
            info = real_fstat(fd)
            return SimpleNamespace(st_size=info.st_size, st_mtime_ns=info.st_mtime_ns,
                st_ctime_ns=info.st_ctime_ns + 42, st_dev=info.st_dev, st_ino=info.st_ino)
        with mock.patch.object(verification.os, "fstat", side_effect=descriptor):
            actual, _ = verification.stable_sha256(self.path)
        self.assertEqual(actual, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_descriptor_ctime_change_during_read_is_not_ignored(self):
        from types import SimpleNamespace
        real_fstat = verification.os.fstat; count = 0
        def descriptor(fd):
            nonlocal count
            info = real_fstat(fd); count += 1
            return SimpleNamespace(st_size=info.st_size, st_mtime_ns=info.st_mtime_ns,
                st_ctime_ns=info.st_ctime_ns + count, st_dev=info.st_dev, st_ino=info.st_ino)
        with mock.patch.object(verification.os, "fstat", side_effect=descriptor):
            with self.assertRaises(verification.FileChangedDuringVerification):
                verification.stable_sha256(self.path)


if __name__ == "__main__": unittest.main()
