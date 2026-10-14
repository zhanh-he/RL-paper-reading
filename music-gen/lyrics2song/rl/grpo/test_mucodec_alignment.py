import math
import unittest


def windows(n, overlap):
    n = max(n, 1024)
    if (n - overlap) % 768:
        n = math.ceil((n - 256) / 768) * 768 + 256
    return [min(1024, n - start) for start in range(0, n - 768, 768)]


class AlignmentTests(unittest.TestCase):
    def test_original_boundary_reproduces_448_latent_frames(self):
        self.assertEqual(windows(1664, 128), [1024, 896])

    def test_corrected_windows_are_complete(self):
        for n in range(1, 8193):
            self.assertTrue(all(w == 1024 for w in windows(n, 256)), n)


if __name__ == '__main__':
    unittest.main()
