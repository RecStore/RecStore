import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from python.pytorch.recstore import telemetry


class TelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        telemetry.reset()

    def tearDown(self) -> None:
        telemetry.reset()

    def test_unavailable_when_library_is_missing(self) -> None:
        with mock.patch.dict(
            os.environ, {"RECSTORE_REPORT_LIBRARY": "/nonexistent/libreport.so"}
        ):
            self.assertFalse(telemetry.available())
            self.assertFalse(telemetry.emit("table", "uid", "duration_us", 1.0))
            self.assertIsNone(telemetry.library_path())

    def test_environment_override_is_the_only_candidate(self) -> None:
        with mock.patch.dict(
            os.environ, {"RECSTORE_REPORT_LIBRARY": "/tmp/custom/libreport.so"}
        ):
            self.assertEqual(
                telemetry._candidate_paths(),
                [Path("/tmp/custom/libreport.so")],
            )

    def test_prefers_the_newest_build_directory(self) -> None:
        def candidates_for(newer: Path, older: Path) -> list[Path]:
            for path, stamp in ((newer, 2_000), (older, 1_000)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
                os.utime(path, (stamp, stamp))
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                telemetry, "_repo_root", return_value=newer.parents[2]
            ):
                return telemetry._candidate_paths()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build" / "lib" / telemetry._LIBRARY_NAME
            release = root / "build_release" / "lib" / telemetry._LIBRARY_NAME
            candidates = candidates_for(release, build)
            self.assertLess(candidates.index(release), candidates.index(build))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build" / "lib" / telemetry._LIBRARY_NAME
            release = root / "build_release" / "lib" / telemetry._LIBRARY_NAME
            candidates = candidates_for(build, release)
            self.assertLess(candidates.index(build), candidates.index(release))

    def test_emit_calls_the_reported_symbol(self) -> None:
        library = mock.MagicMock()
        library.report.return_value = True
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            telemetry, "_candidate_paths", return_value=[Path(__file__)]
        ), mock.patch.object(telemetry.ctypes, "CDLL", return_value=library):
            self.assertTrue(telemetry.emit("table", "uid", "duration_us", 1.5))

        library.report.assert_called_once_with(
            b"table", b"uid", b"duration_us", 1.5
        )

    def test_emit_propagates_false_from_the_library(self) -> None:
        library = mock.MagicMock()
        library.report.return_value = False
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            telemetry, "_candidate_paths", return_value=[Path(__file__)]
        ), mock.patch.object(telemetry.ctypes, "CDLL", return_value=library):
            self.assertFalse(telemetry.emit("table", "uid", "duration_us", 1.0))

    def test_reset_forgets_the_loaded_library(self) -> None:
        library = mock.MagicMock()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            telemetry, "_candidate_paths", return_value=[Path(__file__)]
        ), mock.patch.object(telemetry.ctypes, "CDLL", return_value=library):
            self.assertTrue(telemetry.available())
            self.assertEqual(telemetry.library_path(), __file__)

        telemetry.reset()
        with mock.patch.dict(
            os.environ, {"RECSTORE_REPORT_LIBRARY": "/nonexistent/libreport.so"}
        ):
            self.assertIsNone(telemetry.library_path())


if __name__ == "__main__":
    unittest.main()
