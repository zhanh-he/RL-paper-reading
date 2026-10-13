import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from collect_objective import aggregate, cell
from generate_objective import cmi_conditions
from score_objective import select_candidates
from run_cluster_shard import bounds
from pack_completed_chunk import pack, verified_archive


class ProtocolTests(unittest.TestCase):
    def test_chunk_archive_is_verified_before_cleanup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ['wsb_000_r0_s831001', 'wsb_000_r1_s831019']
            for ident in ids:
                folder = root / 'songs' / ident
                folder.mkdir(parents=True)
                (folder / 'evaluation_binding.json').write_text(json.dumps({'id': ident}))
                (folder / 'result.json').write_text('{}')
            chunk = root / 'evaluation/chunk_000_001.json'
            chunk.parent.mkdir()
            chunk.write_text(json.dumps({'rows': [{'id': ids[0], 'prompt_index': 0,
                'metrics': {'songbench': {}, 'mulan': {}, 'allmusiccaps': {},
                            'q3o': {'status': 'complete'}}, 'supplementary': {}}]}))
            pack(root, chunk)
            self.assertTrue(verified_archive(root, chunk))
            self.assertFalse((root / 'songs' / ids[0]).exists())
            self.assertEqual(len(json.loads((root / 'generation-receipts.json').read_text())), 2)
            (root / 'archives/chunk_000_001.tar').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                verified_archive(root, chunk)

    def test_cluster_shards_cover_exactly_once(self):
        for size in [192, 121]:
            intervals = [(0, 4), *bounds(4, size, 12)]
            flattened = [i for lo, hi in intervals for i in range(lo, hi)]
            self.assertEqual(flattened, list(range(size)))

    def test_q3o_submission_is_not_a_measured_score(self):
        c = cell('control_overall', 'wsb', q3o_submitted=True)
        self.assertEqual(c['status'], 'pending')
        self.assertIsNone(c['value'])
    def test_frozen_cmi_source(self):
        root = Path(__file__).resolve().parents[3]
        split = json.loads((root / 'platform/site/demos/data/cmi-pref-triple-9to1-split.json').read_text())
        rows = cmi_conditions(root / 'datasets/cmi-pref/cmi_test.jsonl', split)
        self.assertEqual(len(rows), 121)
        self.assertEqual(len({r['condition_id'] for r in rows}), 121)
        broken = json.loads(json.dumps(split))
        broken['test'][0]['condition_id'] = 'wrong'
        with self.assertRaises(ValueError):
            cmi_conditions(root / 'datasets/cmi-pref/cmi_test.jsonl', broken)

    def test_per_selection(self):
        def row(i, j, per, edits, length):
            return {'prompt_index': i, 'candidate_index': j, 'metrics': {'per': {
                'per': per, 'edits': edits, 'reference_phonemes': 10,
                'selected_asr_repeat': 0, 'asr_repeats': [{'phonemes': list(range(length))}]}}}
        rows = [row(0, 0, .2, 2, 9), row(0, 1, .1, 1, 10),
                row(1, 1, .2, 2, 10), row(1, 0, .2, 2, 10)]
        self.assertEqual([r['candidate_index'] for r in select_candidates(rows)], [1, 0])

    def test_not_applicable_vs_resource_pending(self):
        self.assertEqual(cell('per', 'cmi')['status'], 'not-applicable')
        self.assertEqual(cell('control_overall', 'cmi')['status'], 'not-applicable')
        self.assertEqual(cell('control_overall', 'wsb')['status'], 'resource-pending')
        self.assertEqual(cell('songbench_global_avg', 'cmi')['status'], 'pending')

    def test_partial_scores_do_not_inherit_previous_song(self):
        def row(i, edits, phonemes):
            return {'id': f'song-{i}', 'prompt_index': i, 'candidate_index': 0,
                    'seconds': 120, 'audio_sha256': 'a', 'source_flac_sha256': 'b',
                    'metrics': {'songbench': {'songbench_raw': dict.fromkeys(
                        ['Melody', 'Arrangement', 'Musicality', 'Vocal', 'Instrumental', 'Mixing', 'Structure'], 6)},
                        'mulan': {'scores': {'mulan': .3}}, 'allmusiccaps': {'scores': {'allmusiccaps': .4}},
                        'per': {'edits': edits, 'reference_phonemes': phonemes, 'per': edits / phonemes}}}
        rows = [row(0, 1, 10), row(1, 9, 30)]
        rows[0]['supplementary'] = {'scorer': {'sha256': 'c'}, 'audiobox_state_sha256': 'd',
            'songeval': dict.fromkeys(['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness'], 4),
            'audiobox': {'PQ': 8}}
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'evaluation').mkdir()
            (run / 'evaluation/chunk_0.json').write_text(json.dumps({'rows': rows}))
            partial = aggregate(run, 'wsb', 192)
            self.assertIsNone(partial['metrics']['songbench_global_avg']['value'])
            self.assertIsNone(partial['diagnostic_rows'][1]['metrics']['songeval_global_avg'])
            self.assertIsNone(partial['diagnostic_rows'][1]['metrics']['audiobox_pq'])
            complete = aggregate(run, 'wsb', 2)
            self.assertAlmostEqual(complete['metrics']['per']['value'], .25)
            self.assertEqual(complete['metrics']['songeval_global_avg']['status'], 'pending')
            (run / 'evaluation/chunk_duplicate.json').write_text(json.dumps({'rows': rows}))
            with self.assertRaises(ValueError):
                aggregate(run, 'wsb', 2)


if __name__ == '__main__':
    unittest.main()
