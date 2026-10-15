import unittest
from lyric_frontend import fallback_language


class FrontendTest(unittest.TestCase):
    def test_native_korean(self):
        self.assertEqual(fallback_language('\uc190 \ub2ff\uc744 \uc218 \uc5c6\ub294 \uc138\uc0c1 \uc18d'), 'ko')

    def test_separators(self):
        self.assertIsNone(fallback_language('-------------------'))
        self.assertIsNone(fallback_language('.'))
        self.assertEqual(fallback_language('123'), 'en')

    def test_no_silent_alphabet_dropping(self):
        with self.assertRaises(ValueError):
            fallback_language('\u0441\u043b\u043e\u0432\u043e')


if __name__ == '__main__':
    unittest.main()
