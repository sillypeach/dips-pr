"""Portable adapters for the archived path kernels and scalar table evidence.

The kernels are byte-identical frozen sources. This module handles paths,
provenance, independent checking, serialization, and complete-seed reporting.
"""
from __future__ import annotations
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
for _key in THREAD_KEYS:
    os.environ[_key] = '1'
COUNTERS = ('nodes', 'pruned_support', 'pruned_canonical', 'pruned_v',
            'correlation_evaluations', 'screened_u', 'traversals',
            'remaining_eligible_pattern_visits', 'all_eligible_pattern_visits')
LABELS = {('count', 2): 'C2', ('count', 4): 'C4', ('count', 8): 'C8',
          ('count', 12): 'C12', ('length', 5): 'L2--5',
          ('length', 6): 'L3--6', ('shared', 30): 'S30'}
ARMLABELS = {'PATH': 'v+u', 'none': 'None', 'v_only': 'v', 'u_only': 'u',
             'vu_single': 'v+u (1ref)', 'vu_two': 'v+u (2ref)'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def obj_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    def convert(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        if hasattr(x, 'item'):
            return x.item()
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                               default=convert) + '\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def config(root):
    cfg = read(root / 'config.json')
    for relative, digest in cfg['source_sha256'].items():
        require(sha(root / relative) == digest, 'Frozen source changed: ' + relative)
    require(len(cfg['taus']) == 101 and all(math.isclose(t, 10**(-i/50),
            rel_tol=1e-14) for i, t in enumerate(cfg['taus'])), 'Invalid registered lambda grid')
    return cfg


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def kernels(root):
    # A fresh CLI process imports one experiment. Refuse mixed unqualified models.
    target = (root / 'frozen/model.py').resolve()
    if 'model' in sys.modules:
        require(Path(sys.modules['model'].__file__).resolve() == target,
                'Different frozen model already imported; use separate CLI processes')
    else:
        module(target, 'model')
    module(root / 'frozen/uv_only_solver.py', 'uv_only_solver')
    path = module(root / 'frozen/path_staged_solver.py', 'path_staged_solver')
    solver = module(root / 'frozen/ablation_staged_solver.py', 'ablation_staged_solver') \
        if root.name == 'ablation' else path
    checker = module(root / 'frozen/audit/model.py', '_release_independent_audit')
    counter_root = root if root.name == 'pruning' else root.parent / 'pruning'
    config(counter_root)
    counter = module(counter_root / 'frozen/path_dictionary_counts.py', '_release_dictionary_counter')
    return sys.modules['model'], path, solver, checker, counter


def load_dataset(root, cfg, dataset_id, inputs=None):
    import numpy as np
    entries = {e['dataset_id']: e for e in cfg['datasets']}
    require(dataset_id in entries, 'Dataset is not registered for this experiment')
    entry = entries[dataset_id]
    inputs = Path(inputs) if inputs else root.parent / 'synthetic/inputs'
    data_path, folds_path = inputs / 'data' / (dataset_id + '.json'), inputs / 'folds' / (dataset_id + '.json')
    require(sha(data_path) == entry['sha256'], 'Dataset hash mismatch')
    require(sha(folds_path) == entry['folds_sha256'], 'Fold hash mismatch')
    data = read(data_path)
    tr, te = data['train_indices'], data['test_indices']
    require(obj_sha(tr) == entry['training_indices_sha256'], 'Changed training row order')
    require(len(tr) == entry['train_size'] and len(te) == entry['test_size'] and
            sorted(tr + te) == list(range(entry['n'])), 'Invalid outer partition')
    tx, y = [data['transactions'][i] for i in tr], np.asarray(data['y'], float)[tr]
    fitcfg = dict(cfg['config'])
    frac, minimum = fitcfg.pop('support_fraction'), fitcfg.pop('support_minimum')
    fitcfg.update(min_support=max(minimum, math.ceil(frac*len(tr))),
                  max_len=entry['maximum_true_length'])
    provenance = dict(dataset_id=dataset_id, data_sha256=entry['sha256'],
        folds_sha256=entry['folds_sha256'], training_indices_sha256=obj_sha(tr),
        training_rows=len(tr), dictionary_config={k:fitcfg[k] for k in
        ('min_support', 'max_len', 'pattern_space', 'propagate_support')},
        original_protocol_sha256=cfg['original_protocol_sha256'],
        release_config_sha256=sha(root / 'config.json'))
    return entry, tx, y, fitcfg, provenance


def audit_fit(audit, m, tx, y, fitted, cfg):
    """Same independent two-scan checks as the frozen experiment's audit_fit."""
    import numpy as np
    started = time.perf_counter()
    fit = fitted['fit']
    patterns = [audit.Pattern(tuple(p.itemset), ()) for p in fit.patterns]
    X = audit.build_X_from_patterns_on_transactions(tx, patterns)
    w, alpha = np.asarray(fit.params.w), np.asarray(fit.alpha)
    eta = fit.params.b + X @ w
    tids, items = audit.PoissonEN_LCM_SPP._transactions_to_item_tidsets(tx)
    calc = audit.PoissonENGap(cfg['lam'], cfg['kappa'])
    e1 = audit.LCMEnumerator(tids, items, len(y), cfg['min_support'], cfg['max_len'], cfg['pattern_space'])
    scan = e1.dual_scan(alpha, cfg['lam'], cfg['kappa'], top_k=0)
    primal, _, gap = calc.gap(y, eta, w, alpha, full_penalty=scan['penalty'])
    e2 = audit.LCMEnumerator(tids, items, len(y), cfg['min_support'], cfg['max_len'], cfg['pattern_space'])
    residual = calc.build_dual_feasible(y, eta)
    scan2 = e2.dual_scan(residual, cfg['lam'], cfg['kappa'],
                        already_have=[p.itemset for p in fit.patterns], top_k=0)
    kkt = max(scan2['outside_violation']/len(y), audit.kkt_residual(
        X, y, audit.ENParams(fit.params.b, w), cfg['lam'], cfg['kappa']))
    objective = m.poisson_objective(y, fit.eta, w, cfg['lam'], cfg['kappa'])
    consistent = bool(np.isclose(primal, objective, rtol=1e-12, atol=1e-7) and
                      np.isclose(gap, fit.gap, rtol=1e-4, atol=1e-7))
    passed = consistent and gap/len(y) <= 1.01*cfg['gap_tol'] and kkt <= 1.01*cfg['solver_tol']
    return dict(status='passed' if passed else 'failed', gap_per_sample=float(gap/len(y)),
        kkt=float(kkt), objective_gap_consistent=consistent, seconds=time.perf_counter()-started,
        scans=[asdict(e1.stats), asdict(e2.stats)], objective=float(objective),
        role='post-fit verification only; no repair or candidate feedback')


def fit_counters(fitted, dictionary_size, arm):
    fit = fitted['fit']
    searches, rounds = [asdict(s) for s in fit.enum_stats_per_round], fit.round_times
    require(len(searches) == len(rounds), 'Extra traversal or inconsistent round counters')
    total = dict.fromkeys(COUNTERS, 0)
    for search, row in zip(searches, rounds):
        n, s, c, v, u = [search[k] for k in
            ('nodes_visited','pruned_support','pruned_canonical','pruned_v','screened_u')]
        require(all(type(x) is int and x >= 0 for x in (n,s,c,v,u)), 'Bad traversal counter')
        d = n-s-c-v
        require(0 <= u <= d <= dictionary_size and d == row['correlation_evaluations'], 'D-U identity failed')
        require(row['same_round_selection'] is False, 'Unexpected selection traversal')
        for key, value in zip(('nodes','pruned_support','pruned_canonical','pruned_v','screened_u','correlation_evaluations'), (n,s,c,v,u,d)):
            total[key] += value
    total.update(traversals=len(searches), all_eligible_pattern_visits=dictionary_size*len(searches),
        remaining_eligible_pattern_visits=total['correlation_evaluations']-total['screened_u'])
    if not searches:
        require(fitted['path_transfer'].get('null_initialization') is True, 'Only analytic endpoint may have zero traversals')
    if arm in ('none','v_only'):
        require(total['screened_u'] == 0 and all(r.get('ws_removed',0) == r.get('postfiltered_candidates',0) == 0 for r in rounds), 'Disabled u component did work')
    if arm in ('none','u_only'):
        require(total['pruned_v'] == 0, 'Disabled v component did work')
    if arm == 'none':
        require(total['remaining_eligible_pattern_visits'] == total['all_eligible_pattern_visits'], 'None arm missed dictionary opportunities')
    denominator = total['all_eligible_pattern_visits']
    total['full_dictionary_exclusion_rate'] = 1-total['remaining_eligible_pattern_visits']/denominator if denominator else None
    return total, searches, rounds


def path_totals(rows):
    require(len({int(r['point']) for r in rows}) == len(rows), 'Duplicate lambda point')
    M = int(rows[0]['eligible_patterns'])
    for r in rows:
        require(r['verified'] in (True,'True') and r['audit_status'] == 'passed', 'Unverified point')
        require(int(r['eligible_patterns']) == M and M > 0, 'Changed dictionary within path')
        require(all(int(r[k]) >= 0 for k in COUNTERS), 'Negative counter')
        d = int(r['correlation_evaluations'])
        require(d == int(r['nodes'])-int(r['pruned_support'])-int(r['pruned_canonical'])-int(r['pruned_v']), 'D=N-S-V-C failed')
        require(int(r['remaining_eligible_pattern_visits']) == d-int(r['screened_u']), 'Remaining=D-U failed')
        require(int(r['all_eligible_pattern_visits']) == M*int(r['traversals']), 'M*R failed')
        require(0 <= int(r['remaining_eligible_pattern_visits']) <= M*int(r['traversals']), 'Invalid surviving opportunities')
        if int(r['traversals']) == 0:
            require(int(r['point']) == 0, 'Nonanalytic zero-traversal point')
            require(r['full_dictionary_exclusion_rate'] in (None,''), 'Analytic endpoint rate must be NA')
        else:
            require(math.isclose(float(r['full_dictionary_exclusion_rate']),
                1-int(r['remaining_eligible_pattern_visits'])/int(r['all_eligible_pattern_visits']),
                abs_tol=1e-12), 'Per-point rate disagrees with counts')
        require(math.isclose(float(r['tau']), 10**(-int(r['point'])/50), rel_tol=1e-13), 'Changed tau grid')
    result = {k:sum(int(r[k]) for r in rows) for k in COUNTERS}
    denominator = result['all_eligible_pattern_visits']
    result.update(eligible_patterns=M,
        full_dictionary_exclusion_rate=1-result['remaining_eligible_pattern_visits']/denominator if denominator else None,
        fit_seconds=sum(float(r['fit_seconds']) for r in rows),
        audit_seconds=sum(float(r['audit_seconds']) for r in rows),
        complete=len(rows)==101 and {int(r['point']) for r in rows}==set(range(101)))
    return result


def new_output(path):
    path = Path(path).resolve()
    require(not path.exists(), 'Output already exists; choose a fresh directory')
    path.mkdir(parents=True)
    return path


def run_paths(root, args, smoke=False):
    cfg = config(root)
    out = new_output(args.output)
    m, pathsolver, solver, checker, counter = kernels(root)
    if smoke:
        import numpy as np
        rng = np.random.default_rng(421)
        tx = [np.flatnonzero(row).tolist() for row in rng.random((48,4)) < .45]
        y = rng.poisson(np.exp(.5 + .4*np.asarray([0 in t and 1 in t for t in tx]))).astype(float)
        entry = dict(dataset_id='smoke', family='smoke', level=0, seed=421)
        fitcfg = {k:v for k,v in cfg['config'].items() if k not in ('support_fraction','support_minimum')}
        fitcfg.update(min_support=3, max_len=2)
        provenance = dict(dataset_id='smoke', generated_fixture_sha256=obj_sha([tx,y.tolist()]),
            release_config_sha256=sha(root/'config.json'), dictionary_config={k:fitcfg[k] for k in
            ('min_support','max_len','pattern_space','propagate_support')})
        jobs = [(entry, tx, y, fitcfg, provenance)]
        arms = cfg['arms']; indices = [0,1,2]
    else:
        ids = [e['dataset_id'] for e in cfg['datasets']] if args.all else args.dataset
        require(ids, 'Specify --dataset or --all')
        require(len(ids) == len(set(ids)), 'Duplicate dataset IDs')
        jobs = [load_dataset(root,cfg,did,args.inputs) for did in ids]
        arms = cfg['arms'] if args.all else getattr(args,'arm',None) or cfg['display_arms']
        require(set(arms) <= set(cfg['arms']), 'Unknown arm')
        require(len(arms) == len(set(arms)), 'Duplicate arms')
        indices = list(range(101))
    registry = dict(version='portable-path-run-v1', study=root.name,
        release_config_sha256=sha(root/'config.json'), measurement_scope='smoke_prefix' if smoke else 'full_outer_paths',
        source_sha256=cfg['source_sha256'], path_files={}, failures=[],
        environment=dict(python=sys.version.split()[0], numpy=__import__('numpy').__version__,
                         scipy=__import__('scipy').__version__, threads={k:os.environ[k] for k in THREAD_KEYS}))
    write(out/'run.json',registry)
    try:
        for index, (entry, tx, y, fitcfg, provenance) in enumerate(jobs):
            directory = out/entry['dataset_id']; directory.mkdir()
            counted = counter.count_dictionary(tx, **provenance['dictionary_config'])
            require(counted['status']=='complete', 'Incomplete full dictionary count')
            counted.update(provenance=provenance, source_sha256=cfg['source_sha256']['frozen/model.py'])
            write(directory/'dictionary.json',counted)
            cal = pathsolver.lambda_max_exact(tx,y,fitcfg['min_support'],fitcfg['max_len'],fitcfg['pattern_space'])
            require(cal['exact'] and cal['lambda_max'] > 0, 'No positive exact lambda_max')
            write(directory/'calibration.json',cal)
            ordered = arms[index % len(arms):] + arms[:index % len(arms)] if getattr(args,'all',False) else arms
            for arm in ordered:
                target = directory/arm; target.mkdir()
                previous, rows, point_hashes = None, [], {}
                for point in indices:
                    fitconfig = dict(fitcfg,lam=cfg['taus'][point]*cal['lambda_max'])
                    kwargs = dict(previous=previous, mode='PATH', calibration=cal, **cfg['staged_options'])
                    if root.name=='ablation': kwargs['ablation_id']=arm
                    start = time.perf_counter()
                    fitted = solver.fit_path_point(tx,y,fitconfig,**kwargs)
                    fit_seconds = time.perf_counter()-start
                    metrics, searches, rounds = fit_counters(fitted,counted['eligible_patterns'],arm)
                    audit = audit_fit(checker,m,tx,y,fitted,fitconfig)
                    fit = fitted['fit']
                    verified = bool(fit.converged and audit['status']=='passed')
                    row = dict(dataset_id=entry['dataset_id'], family=entry['family'],level=entry['level'],seed=entry['seed'],
                        arm=arm,point=point,tau=cfg['taus'][point],config=fitconfig,eligible_patterns=counted['eligible_patterns'],
                        verified=verified,audit_status=audit['status'],fit_seconds=fit_seconds,audit_seconds=audit['seconds'],
                        independent=audit,search=searches,rounds=rounds,patterns=[list(p.itemset) for p in fit.patterns],
                        w=fit.params.w.tolist(),b=float(fit.params.b),alpha=fit.alpha.tolist(),gap=float(fit.gap),
                        state=fitted['state'],**metrics)
                    name=f'point_{point:03d}.json'; write(target/name,row);point_hashes[name]=sha(target/name)
                    require(verified, f'Fit or independent audit failed: {entry["dataset_id"]}/{arm}/{point}')
                    rows.append(row); previous=fitted['state']
                totals=path_totals(rows)
                record=dict(dataset_id=entry['dataset_id'],family=entry['family'],level=entry['level'],seed=entry['seed'],arm=arm,
                    measurement_scope=registry['measurement_scope'],provenance=provenance,
                    dictionary_sha256=sha(directory/'dictionary.json'),calibration_sha256=sha(directory/'calibration.json'),
                    point_sha256=point_hashes,totals=totals)
                write(target/'path.json',record)
                registry['path_files'][str((target/'path.json').relative_to(out))]=sha(target/'path.json')
                write(out/'run.json',registry)
                print(json.dumps(dict(dataset=entry['dataset_id'],arm=arm,complete=totals['complete'],points=len(rows),verified=True)),flush=True)
    except Exception as error:
        registry['failures'].append(str(error));write(out/'run.json',registry);raise
    return out


def archived_points(root):
    cfg=config(root); manifest=read(root/'evidence/manifest.json')
    require(sha(root/'config.json')==manifest['config_sha256'], 'Changed evidence scientific configuration')
    require(sha(root/'evidence/points.csv')==manifest['point_file_sha256'], 'Changed archived point evidence')
    with (root/'evidence/points.csv').open() as f: rows=list(csv.DictReader(f))
    require(len(rows)==manifest['row_count'], 'Evidence row count mismatch')
    return cfg,rows


def saved_points(root, input_root):
    cfg=config(root); input_root=Path(input_root).resolve();reg=read(input_root/'run.json')
    require(reg['study']==root.name and reg['release_config_sha256']==sha(root/'config.json'), 'Run belongs to different frozen configuration')
    require(reg['source_sha256']==cfg['source_sha256'], 'Run source hash mismatch')
    require(not reg['failures'], 'Run contains failures; no formal table is produced')
    require(reg['measurement_scope']=='full_outer_paths', 'Smoke prefixes are not full-path evidence')
    entries={e['dataset_id']:e for e in cfg['datasets']};rows=[]
    for relative,digest in reg['path_files'].items():
        file=(input_root/relative).resolve(); require(file.is_relative_to(input_root), 'Path escapes result root')
        require(sha(file)==digest, 'Changed path output');rec=read(file)
        entry=entries[rec['dataset_id']];proof=rec['provenance']
        require(proof['data_sha256']==entry['sha256'] and proof['folds_sha256']==entry['folds_sha256'] and proof['release_config_sha256']==sha(root/'config.json'), 'Data provenance mismatch')
        require(proof['original_protocol_sha256']==cfg['original_protocol_sha256'], 'Protocol provenance mismatch')
        expected_dictionary=dict(min_support=max(cfg['config']['support_minimum'],math.ceil(cfg['config']['support_fraction']*entry['train_size'])),
            max_len=entry['maximum_true_length'],pattern_space=cfg['config']['pattern_space'],propagate_support=cfg['config']['propagate_support'])
        require(proof['training_indices_sha256']==entry['training_indices_sha256'] and proof['training_rows']==entry['train_size']
            and proof['dictionary_config']==expected_dictionary, 'Training subset or dictionary rules changed')
        directory=file.parent.parent
        require(sha(directory/'dictionary.json')==rec['dictionary_sha256'] and sha(directory/'calibration.json')==rec['calibration_sha256'], 'Changed dictionary/calibration')
        counted=read(directory/'dictionary.json');require(counted['provenance']==proof and counted['status']=='complete', 'Incomplete/mismatched dictionary provenance')
        require(counted['source_sha256']==cfg['source_sha256']['frozen/model.py'], 'Dictionary solver source changed')
        calibration=read(directory/'calibration.json');require(calibration['exact'] is True and calibration['lambda_max']>0, 'Invalid lambda calibration')
        stats=counted['stats'];require(all(stats[k]==0 for k in ('pruned_v','screened_u','pruned_dual','pruned_topk')), 'Denominator used screening')
        require(counted['eligible_patterns']==stats['nodes_visited']-stats['pruned_support']-stats['pruned_canonical'], 'Count dictionary identity failed')
        local=[]
        for name,digest in rec['point_sha256'].items():
            pointfile=(file.parent/name).resolve();require(pointfile.is_relative_to(file.parent), 'Point path escapes result root')
            require(sha(pointfile)==digest, 'Changed point output');row=read(pointfile)
            require(row['dataset_id']==rec['dataset_id'] and row['arm']==rec['arm'] and row['eligible_patterns']==counted['eligible_patterns'], 'Changed path identity/dictionary')
            expected_config={k:v for k,v in cfg['config'].items() if k not in ('support_fraction','support_minimum')}
            expected_config.update(expected_dictionary,lam=calibration['lambda_max']*float(row['tau']))
            require(row['config']==expected_config, 'Changed fitting configuration')
            local.append(row)
        require(path_totals(local)==rec['totals'], 'Path aggregation mismatch')
        rows.extend(local)
    return cfg,rows


def statistics_rows(cfg,points):
    groups={}; entries={e['dataset_id']:e for e in cfg['datasets']}
    for row in points:
        did,arm=row['dataset_id'],row['arm'];require(did in entries and arm in cfg['arms'], 'Unregistered point identity')
        entry=entries[did]
        require(row['family']==entry['family'] and int(row['level'])==entry['level'] and int(row['seed'])==entry['seed'], 'Changed dataset metadata')
        groups.setdefault((did,arm),[]).append(row)
    paths=[]
    for (did,arm),rows in groups.items():
        entry=entries[did]; totals=path_totals(rows)
        paths.append(dict(dataset_id=did,arm=arm,family=entry['family'],level=entry['level'],seed=entry['seed'],**totals))
    condition_keys=list(dict.fromkeys((e['family'],e['level']) for e in cfg['datasets']))
    result=[]
    for family,level in condition_keys:
        expected={e['dataset_id'] for e in cfg['datasets'] if (e['family'],e['level'])==(family,level)}
        require(len(expected)==5, 'Every formal condition must register five seeds')
        complete={arm:[r for r in paths if r['family']==family and r['level']==level and r['arm']==arm and r['complete']] for arm in cfg['arms']}
        gate=all({r['dataset_id'] for r in complete[arm]}==expected for arm in cfg['display_arms'])
        for arm in cfg['arms']:
            rs=complete[arm]; valid=gate and {r['dataset_id'] for r in rs}==expected
            row=dict(condition=LABELS[(family,level)],family=family,level=level,arm=arm,configuration=ARMLABELS[arm],
                     complete_seeds=len(rs),status='complete' if valid else 'incomplete_five_seed_condition')
            for metric,factor in [('eligible_patterns',1),('full_dictionary_exclusion_rate',100),('nodes',1e-6),('fit_seconds',1)]:
                values=[r[metric]*factor for r in rs]
                row[metric+'_mean']=statistics.mean(values) if valid else None
                row[metric+'_sample_sd']=statistics.stdev(values) if valid else None
            result.append(row)
    return paths,result


def report(root,cfg,points,output):
    output=new_output(output);paths,rows=statistics_rows(cfg,points)
    selected=[r for r in rows if r['arm'] in cfg['display_arms']]
    name='table4' if root.name=='ablation' else 'table3'
    write(output/(name+'.json'),selected);write(output/'all_arms.json',rows);write(output/'seed_paths.json',paths)
    with (output/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    def cell(row,key,digits):
        if row[key+'_mean'] is None:return 'NA'
        return f"{row[key+'_mean']:.{digits}f} \\pm {row[key+'_sample_sd']:.{digits}f}"
    lines=[]
    for row in selected:
        cells=[row['condition']]
        if root.name=='ablation':cells += [row['configuration'],cell(row,'full_dictionary_exclusion_rate',2),cell(row,'nodes',2),cell(row,'fit_seconds',1)]
        else:cells += [cell(row,'eligible_patterns',0),cell(row,'full_dictionary_exclusion_rate',2)]
        lines.append(' & '.join(cells)+r' \\')
    (output/(name+'.tex')).write_text('\n'.join(lines)+'\n')
    if root.name=='pruning':
        curves=[]
        for family,level in dict.fromkeys((e['family'],e['level']) for e in cfg['datasets']):
            if not any(r['family']==family and r['level']==level and r['status']=='complete' for r in selected):continue
            for i in range(101):
                rs=[r for r in points if r['family']==family and int(r['level'])==level and int(r['point'])==i]
                require(len(rs)==5,'Incomplete lambda group')
                rates=[100*(1-int(r['remaining_eligible_pattern_visits'])/int(r['all_eligible_pattern_visits'])) for r in rs if int(r['all_eligible_pattern_visits'])>0]
                curves.append(dict(condition=LABELS[(family,level)],point=i,tau=10**(-i/50),
                    mean=statistics.mean(rates) if rates else None,sample_sd=statistics.stdev(rates) if rates else None))
        with (output/'pruning_curves.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['condition','point','tau','mean','sample_sd']);w.writeheader();w.writerows(curves)
    print(json.dumps(dict(output=str(output),table=name,complete_rows=sum(r['status']=='complete' for r in selected),
        total_rows=len(selected),paths=len(paths),scope='complete five-seed conditions only')))
