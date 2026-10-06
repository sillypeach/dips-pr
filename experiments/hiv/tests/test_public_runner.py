"""Portable adapter checks. Numerical fits use invented observations only."""
from pathlib import Path
import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hiv_public_adapter", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def toy_records():
    # Each named group has two invented repeated observations. No HIV records.
    return [[2*g+j, [k for k in range(2) if g & (1 << k)],
             float(2+5*bool(g & 1)+7*bool(g & 2)+j), f"invented-{g:03d}"]
            for g in range(30) for j in range(2)]


def toy_study(root):
    preparer = runner.frozen_preparer()
    data, folds = runner.split_records(toy_records(), 42, preparer)
    data["description"] = "SYNTHETIC_FIXTURE_ONLY; all observations invented"
    expected = dict(preparer.split_fingerprint(data, folds), endpoint="NRTI/AZT",
        outer_split_seed=42, dataset_id="hiv_NRTI_AZT__seed_42", original_dataset_id="hiv_NRTI_AZT")
    entry = runner.write_endpoint(root, dict(expected=expected,data=data,folds=folds,positions=["P1","P2"]), "fixture")
    study = dict(state="frozen",datasets=[entry],primary_selection="CV-min",paper_pr2_key="PseudoR2_train_null")
    study["files"] = {str(p.relative_to(root)):runner.sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    runner.write_new(root/"study.json", study)
    runner.write_new(root/"registered.json",dict(study_sha256=runner.sha(root/"study.json")))
    return entry


class PortableProtocolTests(unittest.TestCase):
    def test_frozen_snapshot_hashes(self):
        runner.verify_files(runner.FROZEN, runner.read(runner.FROZEN/"MANIFEST.json")["files"])

    def test_encoding_missing_response_and_source_row_order(self):
        rows=[{"SeqID":" a ","X":"2.5","P1":" - ","P2":"M"},
              {"SeqID":"b","X":"NA","P1":"Y","P2":"."},
              {"SeqID":"c","X":"nan","P1":"Y","P2":"."},
              {"SeqID":"d","X":"0","P1":"X","P2":" # "}]
        records=runner.endpoint_records(rows,["P1","P2"],"X")
        self.assertEqual(records,[[0,[1],2.5,"a"],[3,[0,1],0.,"d"]])

    def test_grouped_repeats_and_fold_disjointness(self):
        preparer=runner.frozen_preparer()
        memberships=[]
        for seed in runner.SEEDS:
            data,folds=runner.split_records(toy_records(),seed,preparer)
            self.assertFalse(set(data["groups_train"]) & set(data["groups_test"]))
            self.assertEqual(data["y_train"],data["y_final"])
            for fit,valid in folds["folds"]:
                self.assertFalse({data['groups_train'][i] for i in fit} & {data['groups_train'][i] for i in valid})
            self.assertEqual(folds['model_seed'],42)
            self.assertEqual(folds['cv_seed'],seed)
            memberships.append(tuple(sorted(set(data['groups_test']))))
        self.assertEqual(len(set(memberships)),5)

    def test_changed_source_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/"NRTI_DataSet.txt"
            source.write_text("invented nonmatching input")
            with self.assertRaisesRegex(ValueError,"Source snapshot differs"):
                runner.read_class(source,runner.read(runner.FIXTURES/"raw_files.json")[source.name])

    def test_fingerprints_detect_response_and_fold_changes(self):
        preparer=runner.frozen_preparer()
        data,folds=runner.split_records(toy_records(),42,preparer)
        before=preparer.split_fingerprint(data,folds)
        changed=copy.deepcopy(data);changed['y_test'][0]+=1
        self.assertNotEqual(before['data_core_sha256'],preparer.split_fingerprint(changed,folds)['data_core_sha256'])
        changed=copy.deepcopy(folds);changed['folds'][0][0].reverse()
        self.assertNotEqual(before['folds_core_sha256'],preparer.split_fingerprint(data,changed)['folds_core_sha256'])

    def test_help_does_not_require_numeric_import_or_train(self):
        result=subprocess.run([sys.executable,str(HERE/'run.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('validate-inputs',result.stdout)

    def test_training_requires_explicit_target_and_method(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);toy_study(root)
            with patch.object(runner,'call_worker') as worker:
                with self.assertRaisesRegex(ValueError,'--endpoint'):
                    runner.run(root)
                with self.assertRaisesRegex(ValueError,'--method'):
                    runner.run(root,'NRTI/AZT',42)
                worker.assert_not_called()

    def test_changed_registration_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);entry=toy_study(root)
            (root/entry['root']/'data.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                runner.verify_study(root)

    def test_prior_baseline_attempt_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);entry=toy_study(root)
            attempt=root/entry['root']/'baselines/Ridge/request.json'
            runner.write_new(attempt,dict(status='fixture_failure'))
            with patch.object(runner,'call_worker') as worker:
                with self.assertRaisesRegex(ValueError,'previous Ridge attempt'):
                    runner.run(root,'NRTI/AZT',42,'Ridge')
                worker.assert_not_called()

    def test_tiny_ridge_and_full_404_point_dips_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);entry=toy_study(root);eroot=root/entry['root']
            for method in ('Ridge','DIPS-PR'):
                result=subprocess.run([sys.executable,str(HERE/'run.py'),'run','--root',str(root),
                    '--endpoint','NRTI/AZT','--seed','42','--method',method],capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout[-1500:]+result.stderr[-3500:])
            result=runner.read(eroot/'baselines/Ridge/result.json')
            self.assertEqual(result['status'],'complete');self.assertEqual(result['n_candidates'],4)
            for split in ('fold0','fold1','fold2','final'):
                completion=runner.read(eroot/'dips'/split/'completion.json')
                self.assertEqual(completion['status'],'complete')
                self.assertEqual(completion['verified_points'],101)
            selection=runner.read(eroot/'dips/selection.json')
            self.assertFalse(selection['test_used']);self.assertFalse(selection['truth_used'])
            self.assertIn('cv_one_se',runner.read(eroot/'dips/final/selected_results.json'))


if __name__=='__main__':
    unittest.main()
