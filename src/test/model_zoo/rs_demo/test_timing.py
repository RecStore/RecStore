from __future__ import annotations

import unittest
from unittest import mock

from model_zoo.rs_demo.runtime.timing import StepTimer


def _make_cuda_torch():
    """Build a mock torch module with cuda.Event and cuda.synchronize."""
    torch = mock.MagicMock()
    torch.cuda.Event = mock.MagicMock()
    torch.cuda.synchronize = mock.MagicMock()
    return torch


def _make_cuda_device():
    device = mock.Mock()
    device.type = "cuda"
    return device


def _make_cpu_device():
    device = mock.Mock()
    device.type = "cpu"
    return device


class TestStepTimerModes(unittest.TestCase):
    def test_stage_mode_finish_returns_zero_on_cpu(self) -> None:
        row: dict = {}
        timer = StepTimer(row, mock.MagicMock(), _make_cpu_device(), mode="stage")
        self.assertEqual(timer.finish(), 0.0)

    def test_step_mode_finish_returns_zero_on_cpu(self) -> None:
        row: dict = {}
        timer = StepTimer(row, mock.MagicMock(), _make_cpu_device(), mode="step")
        self.assertEqual(timer.finish(), 0.0)

    def test_none_mode_finish_returns_zero_on_cpu(self) -> None:
        row: dict = {}
        timer = StepTimer(row, mock.MagicMock(), _make_cpu_device(), mode="none")
        self.assertEqual(timer.finish(), 0.0)

    def test_none_mode_gpu_falls_back_to_wall_clock(self) -> None:
        row: dict = {}
        timer = StepTimer(row, mock.MagicMock(), _make_cpu_device(), mode="none")
        with timer.gpu("key"):
            pass
        self.assertIn("key", row)
        self.assertGreaterEqual(row["key"], 0.0)

    def test_invalid_mode_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid timing sync mode"):
            StepTimer({}, mock.MagicMock(), _make_cpu_device(), mode="bogus")

    def test_none_mode_skips_synchronize_on_cuda(self) -> None:
        torch_mock = _make_cuda_torch()
        timer = StepTimer({}, torch_mock, _make_cuda_device(), mode="none")
        self.assertEqual(timer.finish(), 0.0)
        torch_mock.cuda.synchronize.assert_not_called()

    def test_stage_mode_calls_synchronize_on_cuda(self) -> None:
        torch_mock = _make_cuda_torch()
        timer = StepTimer({}, torch_mock, _make_cuda_device(), mode="stage")
        timer.finish()
        torch_mock.cuda.synchronize.assert_called_once()

    def test_step_mode_calls_synchronize_on_cuda(self) -> None:
        torch_mock = _make_cuda_torch()
        timer = StepTimer({}, torch_mock, _make_cuda_device(), mode="step")
        timer.finish()
        torch_mock.cuda.synchronize.assert_called_once()

    def test_step_mode_gpu_falls_back_to_wall_clock(self) -> None:
        row: dict = {}
        torch_mock = _make_cuda_torch()
        timer = StepTimer(row, torch_mock, _make_cuda_device(), mode="step")
        with timer.gpu("key"):
            pass
        # In step mode, gpu() uses wall-clock, so no CUDA events recorded.
        torch_mock.cuda.Event.assert_not_called()
        self.assertIn("key", row)
        self.assertGreaterEqual(row["key"], 0.0)

    def test_none_mode_gpu_does_not_record_events_on_cuda(self) -> None:
        row: dict = {}
        torch_mock = _make_cuda_torch()
        timer = StepTimer(row, torch_mock, _make_cuda_device(), mode="none")
        with timer.gpu("key"):
            pass
        torch_mock.cuda.Event.assert_not_called()
        self.assertIn("key", row)

    def test_stage_mode_gpu_records_events_on_cuda(self) -> None:
        row: dict = {}
        torch_mock = _make_cuda_torch()
        timer = StepTimer(row, torch_mock, _make_cuda_device(), mode="stage")
        with timer.gpu("key"):
            pass
        # stage mode on CUDA creates 2 events (start + end).
        self.assertEqual(torch_mock.cuda.Event.call_count, 2)


if __name__ == "__main__":
    unittest.main()
