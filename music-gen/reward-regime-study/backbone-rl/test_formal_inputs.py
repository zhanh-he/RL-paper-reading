import json
from pathlib import Path
import tempfile
import unittest
from formal_inputs import read_jsonl


class PhysicalJSONL(unittest.TestCase):
    def test_unicode_lyric_separators_do_not_change_record_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            rows = [{"messages": [{"role": "user", "content": "verse\u2028chorus\u0085end"}]}] * 3
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            self.assertGreater(len(path.read_text().splitlines()), 3)
            self.assertEqual(read_jsonl(path), rows)


if __name__ == "__main__":
    unittest.main()
