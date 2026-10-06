"""Fast archive/provenance tests. No paper-scale fitting or server access."""
import copy
import importlib.util
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('release_path_reporting_tests',ROOT/'common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)

class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.points=common.archived_points(ROOT)
        cls.ablation_config,cls.ablation_points=common.archived_points(ROOT.parent/'ablation')

    def test_shared_inputs_match_registered_training_provenance(self):
        for root,cfg in [(ROOT,self.config),(ROOT.parent/'ablation',self.ablation_config)]:
            entry,tx,y,fitcfg,proof=common.load_dataset(root,cfg,'count_02_s2609271101')
            self.assertEqual(len(tx),1400)
            self.assertEqual(len(y),1400)
            self.assertEqual(fitcfg['min_support'],21)
            self.assertEqual(fitcfg['max_len'],4)
            self.assertEqual(proof['training_indices_sha256'],entry['training_indices_sha256'])

    def test_table3_published_values(self):
        paths,rows=common.statistics_rows(self.config,self.points)
        self.assertEqual(len(paths),35)
        expected=[('C2','51127','1513','96.20','0.45'),('C4','154293','3951','95.41','0.51'),
          ('C8','417819','7210','87.94','2.18'),('C12','617017','4957','85.18','1.37'),
          ('L2--5','201389','7483','95.71','1.01'),('L3--6','1155132','58720','95.12','0.69'),
          ('S30','766428','35118','80.50','5.35')]
        actual=[(r['condition'],f"{r['eligible_patterns_mean']:.0f}",f"{r['eligible_patterns_sample_sd']:.0f}",
                 f"{r['full_dictionary_exclusion_rate_mean']:.2f}",f"{r['full_dictionary_exclusion_rate_sample_sd']:.2f}") for r in rows]
        self.assertEqual(actual,expected)

    def test_table4_published_values(self):
        paths,rows=common.statistics_rows(self.ablation_config,self.ablation_points)
        self.assertEqual(len(paths),75)
        actual=[(r['condition'],r['arm'],*[f"{r[k]:.{d}f}" for k,d in [
            ('full_dictionary_exclusion_rate_mean',2),('full_dictionary_exclusion_rate_sample_sd',2),
            ('nodes_mean',2),('nodes_sample_sd',2),('fit_seconds_mean',1),('fit_seconds_sample_sd',1)]])
            for r in rows if r['arm'] in self.ablation_config['display_arms']]
        expected=[('C2','v_only','86.46','0.81','17.03','2.17','206.1','15.2'),
            ('C2','u_only','96.20','0.45','77.60','9.72','548.6','14.4'),
            ('C2','vu_two','96.20','0.45','17.02','2.15','286.8','18.7'),
            ('C4','v_only','89.42','0.94','29.13','5.23','580.1','65.1'),
            ('C4','u_only','95.41','0.51','148.74','14.89','1735.7','49.2'),
            ('C4','vu_two','95.41','0.51','28.23','5.07','787.8','85.7'),
            ('L2--5','v_only','90.54','1.66','56.68','20.70','689.5','140.6'),
            ('L2--5','u_only','95.71','1.01','404.42','89.41','2378.0','96.8'),
            ('L2--5','vu_two','95.71','1.01','55.27','20.31','952.1','212.4')]
        self.assertEqual(actual,expected)

    def test_missing_seed_suppresses_whole_displayed_condition(self):
        rows=[p for p in self.ablation_points if not (p['dataset_id']=='count_02_s2609271105' and p['arm']=='vu_two')]
        _,summary=common.statistics_rows(self.ablation_config,rows)
        for r in summary:
            if r['condition']=='C2':
                self.assertEqual(r['status'],'incomplete_five_seed_condition')
                self.assertIsNone(r['full_dictionary_exclusion_rate_mean'])
            else:self.assertEqual(r['status'],'complete')

    def test_partial_path_is_not_formal(self):
        rows=[p for p in self.points if not (p['dataset_id']=='count_02_s2609271101' and p['point']=='100')]
        _,summary=common.statistics_rows(self.config,rows)
        self.assertEqual(summary[0]['complete_seeds'],4)
        self.assertIsNone(summary[0]['full_dictionary_exclusion_rate_mean'])

    def test_counter_corruptions_rejected(self):
        path=[p for p in self.points if p['dataset_id']=='count_02_s2609271101']
        for key,value in [('remaining_eligible_pattern_visits','-1'),('all_eligible_pattern_visits','1'),
                          ('nodes','0'),('tau','0.5'),('verified','False'),('audit_status','failed'),
                          ('full_dictionary_exclusion_rate','0.123')]:
            bad=copy.deepcopy(path);bad[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):common.path_totals(bad)
        bad=copy.deepcopy(path);bad[0]['full_dictionary_exclusion_rate']='1.0'
        with self.assertRaises(ValueError):common.path_totals(bad)
        with self.assertRaises(ValueError):common.path_totals(path+[path[0]])

    def test_path_ratio_is_count_weighted(self):
        path=[p for p in self.points if p['dataset_id']=='count_02_s2609271101']
        ratio=common.path_totals(path)['full_dictionary_exclusion_rate']
        simple_mean=statistics.mean(float(p['full_dictionary_exclusion_rate']) for p in path if p['full_dictionary_exclusion_rate'])
        self.assertGreater(abs(ratio-simple_mean),0.001)

if __name__=='__main__':unittest.main()
