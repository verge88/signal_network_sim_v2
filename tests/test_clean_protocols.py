"""The v2 GUI exposes only supported signalling protocols."""
import unittest

from gui.model import PROTOCOLS, template


class CleanProtocolTests(unittest.TestCase):
    def test_supported_protocols(self):
        self.assertEqual(set(PROTOCOLS), {"ss7", "diameter", "sip"})

    def test_no_5g_template(self):
        with self.assertRaises(ValueError):
            template("5g_sba")


if __name__ == "__main__":
    unittest.main()
