import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("check_test_registration.py")


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_test_registration", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckTestRegistrationTest(unittest.TestCase):
    def test_commented_out_registration_does_not_count(self):
        checker = _load_checker()
        text = checker._strip_cmake_comments(
            "add_executable(test_a a.cpp)\n"
            "# add_executable(test_b b.cpp)\n"
        )
        self.assertTrue(checker._registered(text, "test_a"))
        self.assertFalse(checker._registered(text, "test_b"))


if __name__ == "__main__":
    unittest.main()
