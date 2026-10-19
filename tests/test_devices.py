import unittest

from object_detection.devices import select_device
from object_detection.errors import ConfigurationError


class _Available:
    def __init__(self, value: bool) -> None:
        self.value = value

    def is_available(self) -> bool:
        return self.value


class _Torch:
    def __init__(self, cuda: bool, mps: bool) -> None:
        self.cuda = _Available(cuda)
        self.backends = type("Backends", (), {"mps": _Available(mps)})()


class DeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self) -> None:
        self.assertEqual(select_device("auto", _Torch(True, True)), "cuda")
        self.assertEqual(select_device("auto", _Torch(False, True)), "mps")
        self.assertEqual(select_device("auto", _Torch(False, False)), "cpu")

    def test_unavailable_explicit_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "CUDA"):
            select_device("cuda", _Torch(False, False))

    def test_invalid_device_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            select_device("tpu", _Torch(False, False))


if __name__ == "__main__":
    unittest.main()

