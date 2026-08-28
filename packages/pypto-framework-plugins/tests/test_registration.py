"""Reversible CUDA registration transaction tests (CPU, CUDA hidden)."""

from __future__ import annotations

import unittest
from unittest import mock

from pypto_plugins.torch import registration
from pypto_plugins.torch.context import activate_mode
import pypto_plugins.torch_inductor as torch_inductor


class RegistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from torch._inductor.codegen import common

        common.init_backend_registration()
        cls.common = common

    def test_install_captures_original_and_is_reversible(self) -> None:
        original = self.common.device_codegens["cuda"]
        snapshot = registration.install()
        self.assertTrue(registration.installed())
        self.assertIs(snapshot.scheduling, original.scheduling)
        self.assertIsNot(self.common.device_codegens["cuda"], original)
        registration.uninstall()
        self.assertFalse(registration.installed())
        self.assertIs(self.common.device_codegens["cuda"].scheduling, original.scheduling)

    def test_dispatch_delegates_outside_and_fails_closed_inside(self) -> None:
        registration.install()
        self.addCleanup(registration.uninstall)
        scheduling_ctor = self.common.device_codegens["cuda"].scheduling
        from torch._inductor.codegen.cuda_combined_scheduling import (
            CUDACombinedScheduling,
        )

        scheduler = scheduling_ctor(object())
        self.assertIsInstance(scheduler, CUDACombinedScheduling)
        with activate_mode(strict=True):
            with self.assertRaises(Exception):
                scheduling_ctor(object()).codegen()
        self.assertIsNotNone(scheduler)

    def test_pypto_wrapper_disables_unrestorable_inductor_disk_cache(self) -> None:
        registration.install()
        self.addCleanup(registration.uninstall)
        wrapper = self.common.device_codegens["cuda"].wrapper_codegen
        with activate_mode(strict=True):
            self.assertFalse(wrapper.supports_caching)

    def test_public_install_contract_uses_real_registration(self) -> None:
        torch_inductor.uninstall()
        self.addCleanup(torch_inductor.uninstall)
        with (
            mock.patch.object(torch_inductor, "assert_torch_compatible"),
            mock.patch.object(torch_inductor, "prepare_process_strict"),
        ):
            torch_inductor.install()
        self.assertTrue(torch_inductor._INSTALLED)
        self.assertTrue(registration.installed())


if __name__ == "__main__":
    unittest.main()
