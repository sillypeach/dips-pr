"""Independently check complete grouped HIV results and produce a paired report."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

def load(p):
    return json.loads(p.read_text())

def metrics(y, pr, fitmean):
    from scipy.special import xlogy
    p=np.asarray(pr,float)
    d=lambda mu:max(0.,float(2*np.sum(xlogy(y,y/np.maximum(mu,1e-6))-y+np.maximum(mu,1e-6))))
    dev=d(np.maximum(p,1e-6))
    sst=float(np.sum((y-y.mean())**2)); null=d(np.full_like(y,y.mean())); null_fit=d(np.full_like(y,fitmean))
    return dict(R2=float(1-np.sum((y-p)**2)/sst) if sst else None,
        RMSE=float(np.sqrt(np.mean((y-p)**2))),MAE=float(np.mean(abs(y-p))),
        PseudoR2=1-dev/null if null else None,
        PseudoR2_train_null=1-dev/null_fit if null_fit else None)


def equivalent_metric(value, saved):
    return saved is None if value is None else saved is not None and np.isclose(value,saved,rtol=1e-10,atol=1e-8)


def fmt(value):
    return 'NA' if value is None else f'{value:.4f}'

def export_patterns(root,data,final,folds,choice,protocol):
    import math
    mapping=load(root/'feature_mapping.json')
    items=mapping['items']
    byid={row['item_id']:row for row in items}
    out=root/'analysis';out.mkdir(exist_ok=True)
    reports={}
    for label,key in [('CV-min','min_index'),('1SE','one_se_index')]:
        index=choice[key];row=final[index]
        patterns=row['patterns'];weights=row['w']
        assert len(patterns)==len(weights)
        tx=[set(t) for t in data['tx_final']]
        folds_selected=[{tuple(sorted(pat)) for pat,w in zip(f[index]['patterns'],f[index]['w']) if abs(w)>1e-8} for f in folds]
        terms=[]
        for pattern,weight in zip(patterns,weights):
            if abs(weight)<=1e-8:continue
            normalized=tuple(sorted(pattern))
            columns=[byid[item]['column'] for item in normalized]
            support=sum(set(normalized)<=t for t in tx)
            terms.append(dict(item_ids=list(normalized),columns=columns,length=len(pattern),coefficient=weight,
                term_multiplier=math.exp(weight),training_support=support,training_fraction=support/len(tx),
                cv_fold_presence=sum(normalized in fs for fs in folds_selected),cv_folds=3))
        terms.sort(key=lambda t:(-abs(t['coefficient']),t['item_ids']))
        assert len(terms)==row['NZ']
        test_tx=[set(t) for t in data['tx_test']]
        pr=[math.exp(row['b']+sum(w for pat,w in zip(patterns,weights) if set(pat)<=t)) for t in test_tx]
        assert np.allclose(pr,row['prediction'],rtol=1e-10,atol=1e-8)
        reports[label]=dict(tau=row['tau'],lam=row['config']['lam'],NZ=row['NZ'],intercept=row['b'],terms=terms)
    output=dict(endpoint=protocol['endpoint'],encoding=mapping.get('encoding'),models=reports,
        notes=['All nonzero terms are retained. Sort uses absolute fitted coefficient, never test effects.',
               'CV fold presence uses the same globally CV-selected tau; it is descriptive, not selection.',
               'Positions do not identify amino-acid substitutions. Fitted associations are not causal effects.',
               'exp(coefficient) describes one term contribution; overlapping terms can activate together.'])
    (out/'selected_patterns.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# Selected position combinations: '+protocol['endpoint'],'',
        'CV-min is primary. All nonzero terms are listed; presence indicators do not distinguish substitutions.',
        'Coefficients are penalized conditional associations. Fold recurrence is descriptive and did not select terms.','']
    for name,model in reports.items():
        lines+=['## '+name,'',f"NZ={model['NZ']}; tau={model['tau']:.6g}; lambda={model['lam']:.6g}.",'',
          '|Position combination|Coefficient|exp(coefficient)|Training support|Fold recurrence|',
          '|---|---:|---:|---:|---:|']
        lines += ['|'+ ' & '.join(t['columns'])+f"|{t['coefficient']:.6f}|{t['term_multiplier']:.5f}|{t['training_support']}/{len(tx)}|{t['cv_fold_presence']}/3|" for t in model['terms']]
        lines+=['']
    (out/'SELECTED_PATTERNS.md').write_text('\n'.join(lines)+'\n')


def run(root):
    root=Path(root).resolve()
    sys.path.insert(0,str(root/'source'))
    spec=importlib.util.spec_from_file_location('azt_verify',root/'source/hiv_dips.py')
    worker=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    p=worker.check(root)
    folds=[worker.verify_split(root,p,f'fold{i}') for i in range(3)]
    final=worker.verify_split(root,p,'final')
    choice=load(root/'dips/selection.json')
    expected=worker.select_scores([[r['metrics']['RMSE'] for r in f] for f in folds],p['taus'])
    for k,v in expected.items():
        assert choice[k]==v,k
    data=load(root/'data.json')
    saved_folds=load(root/'folds.json')['folds']
    bs=importlib.util.spec_from_file_location('azt_baseline_verify',root/'source/hiv_baseline.py')
    baseline_worker=importlib.util.module_from_spec(bs)
    bs.loader.exec_module(baseline_worker)
    baseline_worker.verify_registered(root)
    specs=baseline_worker.load_frozen_specs(root,p['model_seed'])
    from sklearn.model_selection import ParameterGrid
    normalized=lambda v:json.loads(json.dumps(v))
    y=np.asarray(data['y_test'],float)
    fitmean=float(np.mean(data['y_final']))
    records=[]
    for name,key in [('DIPS-PR (CV-min)','min_index'),('DIPS-PR (1SE)','one_se_index')]:
        row=final[choice[key]]
        score=metrics(y,row['prediction'],fitmean)
        for k,v in score.items():
            assert equivalent_metric(v,row['metrics'][k]),(name,k)
        records.append(dict(method=name,metrics=score,tau=row['tau'],lam=row['config']['lam'],
                            NZ=row['NZ'],cv_rmse=choice['mean_rmse'][choice[key]],
                            max_prediction=max(row['prediction']),warnings=len(row.get('warnings',[]))))
    hashes={}
    for name in p['baselines']:
        folder=root/'baselines'/name
        result=load(folder/'result.json')
        assert result['status']=='complete'
        assert result['protocol_sha256']==hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
        rows=load(folder/'cv_results.json')['candidates']
        cv_warning_count=0
        grid=normalized(list(ParameterGrid(specs[name][1])))
        assert len(rows)==len(grid)
        assert [r['candidate_index'] for r in rows]==list(range(len(grid)))
        for row in rows:
            assert row['params']==grid[row['candidate_index']]
            losses=[]; fold_valid=[]
            for k in range(3):
                f=load(folder/f'candidate_{row["candidate_index"]:03d}/fold_{k}.json')
                cv_warning_count+=len(f.get('warnings',[]))
                assert f['candidate_index']==row['candidate_index'] and f['fold_index']==k
                assert [f['train_indices'],f['validation_indices']]==saved_folds[k]
                assert f['params']==row['params']
                fold_valid.append(f['valid'])
                if f['valid']:
                    va=saved_folds[k][1]
                    truth=np.asarray(data['y_train'])[va]
                    loss=float(np.sqrt(np.mean((truth-np.asarray(f['prediction']))**2)))
                    assert np.isclose(loss,f['rmse'],rtol=1e-11,atol=1e-9)
                    losses.append(loss)
            assert row['valid']==all(fold_valid)
            if row['valid']:
                assert len(losses)==3
                assert np.isclose(np.mean(losses),row['mean_rmse'],rtol=1e-11,atol=1e-9)
        best=min([r for r in rows if r['valid']],key=lambda r:(r['mean_rmse'],r['candidate_index']))
        assert best['candidate_index']==result['best_candidate_index']
        fit=load(folder/'final_fit.json')
        assert result['prediction']==fit['prediction']
        assert result['best_params']==best['params']==fit['params']
        assert fit['valid'] and fit['n_fit']==len(data['y_final']) and fit['n_predict']==len(y)
        assert np.isclose(best['mean_rmse'],result['cv_rmse'],rtol=1e-12,atol=1e-10)
        assert result['valid_candidate_count']==sum(r['valid'] for r in rows)
        assert result['invalid_candidate_count']==sum(not r['valid'] for r in rows)
        score=metrics(y,result['prediction'],fitmean)
        for k,v in score.items():
            assert equivalent_metric(v,result['metrics'][k]),(name,k)
        records.append(dict(method=name,metrics=score,cv_rmse=result['cv_rmse'],
            max_prediction=max(result['prediction']),warnings=len(result['warnings']),
            final_warnings=len(result['warnings']),cv_warnings=cv_warning_count,
            invalid_candidates=result['invalid_candidate_count']))
        hashes[str((folder/'result.json').relative_to(root))]=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest()
    export_patterns(root,data,final,folds,choice,p)
    summary=dict(endpoint=p['endpoint'], paper_pr2_key='PseudoR2_train_null', status='independently_verified',path_points=404,baseline_methods=len(p['baselines']),
                 path_warning_count=sum(len(r.get('warnings',[])) for path in folds+[final] for r in path),
                 selection=choice,rows=records,baseline_result_hashes=hashes)
    out=root/'analysis';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=['# HIV SeqID 分组重训结果','',
          '外层与三折CV均按SeqID分组。CV-min为主结果，1SE为预设补充；测试集未用于选参。主PR²使用拟合集均值作为null。此数据此前已被分析，不宣称外部验证。', '',
          '|方法|CV RMSE|Test R²|Poisson PR²|RMSE|MAE|NZ|',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['metrics']
        text.append(f'|{r["method"]}|{r["cv_rmse"]:.4f}|{fmt(m["R2"])}|{fmt(m["PseudoR2_train_null"])}|{m["RMSE"]:.4f}|{m["MAE"]:.4f}|{r.get("NZ","—")}|')
    text+=['','404个DIPS路径点完成全字典审计；三折选参与基线CV分数、最终预测指标已独立复算。',
           '详细协议、原始警告与失败候选保留在同目录。没有删除测试样本或裁剪正式预测。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status=summary['status'],rows=records),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    run(p.parse_args().root)
