"""Basic initialization test."""

import unittest


class TestBasic(unittest.TestCase):
    def test_import(self) -> None:
        import seqsm

        self.assertEqual(seqsm.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
