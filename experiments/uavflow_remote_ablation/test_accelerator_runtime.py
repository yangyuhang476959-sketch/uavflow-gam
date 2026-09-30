from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from experiments.uavflow_predictor_idm.runtime import (
    amp_dtype,
    autocast_context,
    configure_accelerator,
    rand_on_device,
    requested_accelerator,
    step_generator,
)


class AcceleratorRuntimeTests(unittest.TestCase):
    def test_accelerator_aliases(self):
        for value, expected in (
            ("gpu", "cuda"),
            ("ascend", "npu"),
            ("NPU", "npu"),
            ("auto", "auto"),
        ):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"UAVFLOW_ACCELERATOR": value}, clear=False
            ):
                self.assertEqual(requested_accelerator(), expected)

    def test_amp_auto_is_backend_specific(self):
        self.assertEqual(amp_dtype("auto", "cuda"), torch.bfloat16)
        self.assertEqual(amp_dtype("auto", "npu"), torch.float16)
        self.assertEqual(amp_dtype("bf16", "npu"), torch.bfloat16)

    def test_cpu_path_does_not_require_torch_npu(self):
        with mock.patch.dict(
            os.environ, {"UAVFLOW_ACCELERATOR": "cpu"}, clear=False
        ):
            accelerator = configure_accelerator(local_rank=0, distributed=False)
            self.assertEqual(accelerator.kind, "cpu")
            self.assertEqual(accelerator.distributed_backend, "gloo")
            with autocast_context(accelerator.device, enabled=True):
                value = torch.ones(2).sum()
            self.assertEqual(float(value), 2.0)

    def test_invalid_accelerator_fails_early(self):
        with mock.patch.dict(
            os.environ, {"UAVFLOW_ACCELERATOR": "not-a-device"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "UAVFLOW_ACCELERATOR"):
                requested_accelerator()

    def test_cpu_generator_preserves_step_randomness(self):
        with mock.patch.dict(
            os.environ, {"UAVFLOW_ACCELERATOR": "cpu"}, clear=False
        ):
            accelerator = configure_accelerator(local_rank=0, distributed=False)
            first = rand_on_device(
                (4,), device="cpu", generator=step_generator(accelerator, 123)
            )
            second = rand_on_device(
                (4,), device="cpu", generator=step_generator(accelerator, 123)
            )
            self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
