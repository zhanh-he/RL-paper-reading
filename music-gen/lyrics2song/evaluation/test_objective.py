import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from collect_objective import aggregate, cell
from generate_objective import cmi_conditions
from score_objective import select_candidates, save_prepared_chunk
from run_cluster_shard import bounds, clean_install_cache
from pack_completed_chunk import pack, verified_archive
from stage_install_cache import stage, finish
from export_quality_inputs import validate_assignment
from balance_evaluator_runtime import balance
from merge_returned_quality import merge
from scorer_cache import needs_scoring


class ProtocolTests(unittest.TestCase):
    def test_failed_receipts_are_archived_and_retried(self):
        with TemporaryDirectory() as directory:
            receipt = Path(directory) / 'songbench.json'
            failed = {'status': 'failed', 'error': 'CUDA OOM'}
            receipt.write_text(json.dumps({'identity': 'frozen', 'result': failed}))
            original = receipt.read_bytes()
            self.assertTrue(needs_scoring(receipt, failed))
            self.assertEqual(receipt.read_bytes(), original)
            archives = list((receipt.parent / 'failed-attempts').glob('*.json'))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), original)
            self.assertTrue(needs_scoring(receipt, failed))
            self.assertEqual(len(list(archives[0].parent.glob('*.json'))), 1)
            self.assertFalse(needs_scoring(receipt, {'status': 'complete'}))
            self.assertTrue(needs_scoring(receipt.parent / 'missing.json', None))

    def test_remote_quality_cannot_change_candidates_or_replace_scores(self):
        rows = [{'id': str(i), 'prompt_index': i//2, 'candidate_index': i%2, 'seed': i,
                 'audio_sha256': str(i), 'source_flac_sha256': str(i), 'adapter_sha256': 'adapter',
                 'seconds': 60, 'target': {}, 'audio_path': 'original', 'metrics': {}}
                for i in range(8)]
        original = {'generation_protocol': {'dataset': 'cmi'}, 'rows': rows}
        returned = json.loads(json.dumps({**original, 'complete': True}))
        for row in returned['rows']:
            row['audio_path'] = 'remote'
            row['metrics'] = {'songbench': {'status': 'complete', 'songbench_raw': {str(i): 3 for i in range(7)}},
                              'mulan': {'status': 'complete', 'scores': {'mulan': 0.3}},
                              'allmusiccaps': {'status': 'complete', 'scores': {'allmusiccaps': 0.3}}}
            row['supplementary'] = {'songeval': {str(i): 3 for i in range(5)}, 'audiobox': {'PQ': 8}}
        self.assertEqual(merge(original, returned)['rows'][0]['audio_path'], 'original')
        returned['rows'][0]['seed'] += 1
        with self.assertRaises(ValueError):
            merge(original, returned)
        returned['rows'][0]['seed'] -= 1
        rows[0]['metrics']['songbench'] = {}
        with self.assertRaises(ValueError):
            merge(original, returned)

    def test_runtime_relocation_preserves_prefix_and_files(self):
        with TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / 'wsb_metrics/.venvs/eval'
            source.mkdir(parents=True)
            (source / 'package').write_text('unchanged installed package')
            (work / 'preparation-receipt.json').write_text(json.dumps({'status': 'prepared'}))
            balance(work)
            self.assertTrue(source.is_symlink())
            self.assertEqual((source / 'package').read_text(), 'unchanged installed package')
            balance(work)

    def test_kaya_production_assignment_retains_both_fixed_seeds(self):
        rows = [{'id': f'cmi_{i:03d}_r{j}_s{[931001, 931019][j]+i}', 'prompt_index': i,
                 'candidate_index': j, 'seed': [931001, 931019][j] + i, 'metrics': {}}
                for i in range(4) for j in range(2)]
        chunk = {'generation_protocol': {'dataset': 'cmi'}, 'rows': rows}
        validate_assignment(chunk)
        rows[0]['seed'] += 1
        with self.assertRaises(ValueError):
            validate_assignment(chunk)
        rows[0]['seed'] -= 1
        rows[0]['metrics']['songbench'] = 4
        with self.assertRaises(ValueError):
            validate_assignment(chunk)

    def test_mp3_preparation_preserves_completed_scores(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'chunk.json'
            row = {'id': 'one', 'audio_sha256': 'mp3', 'adapter_sha256': 'adapter',
                   'source_flac_sha256': 'flac', 'seed': 12}
            previous = {'generation_protocol': {'source': 'frozen'}, 'complete': True,
                        'rows': [{**row, 'metrics': {'songbench': 3.5}}]}
            path.write_text(json.dumps(previous))
            save_prepared_chunk(path, [row], {'source': 'frozen'})
            self.assertEqual(json.loads(path.read_text()), previous)
            with self.assertRaises(ValueError):
                save_prepared_chunk(path, [{**row, 'seed': 13}], {'source': 'frozen'})

    def test_install_cache_uses_jobfs_and_preserves_environments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'work'
            bundle = work / 'wsb_metrics'
            cache = bundle / '.cache/uv'
            cache.mkdir(parents=True)
            (cache / 'wheel').write_text('cache')
            (bundle / '.venvs').mkdir()
            key_root = work / 'key-env'
            stage(bundle, root / 'jobfs', key_root)
            self.assertTrue(cache.is_symlink())
            self.assertEqual((cache / 'wheel').read_text(), 'cache')
            self.assertEqual((bundle / '.venvs/key').resolve(), key_root.resolve())
            with self.assertRaises(FileNotFoundError):
                finish(bundle)
            (work / 'preparation-receipt.json').write_text(json.dumps({'status': 'prepared'}))
            finish(bundle)
            self.assertFalse(cache.is_symlink())
            self.assertTrue((root / 'jobfs/uv/wheel').exists())
            self.assertTrue(key_root.exists())

    def test_cache_cleanup_requires_completed_preparation(self):
        with TemporaryDirectory() as directory:
            work = Path(directory)
            cache = work / 'wsb_metrics/.cache/uv'
            cache.mkdir(parents=True)
            (cache / 'wheel').write_text('download-only cache')
            receipt = work / 'preparation-receipt.json'
            receipt.write_text(json.dumps({'status': 'incomplete'}))
            with self.assertRaises(ValueError):
                clean_install_cache(work)
            self.assertTrue(cache.exists())
            receipt.write_text(json.dumps({'status': 'prepared'}))
            clean_install_cache(work)
            self.assertFalse(cache.exists())

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
