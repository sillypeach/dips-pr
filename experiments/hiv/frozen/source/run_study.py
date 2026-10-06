"""Run the one-shot controller, then verify completed endpoints independently."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import dispatch


def main(root):
    root=Path(root).resolve()
    study,digest=dispatch.verify_study(root)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    result=subprocess.run([sys.executable,str(root/'source/dispatch.py'),'--root',str(root)])
    records=[]
    for entry in study['datasets']:
        endpoint=root/entry['root']
        complete=(endpoint/'dips/final/completion.json').exists() and all(
            (endpoint/'baselines'/method/'result.json').exists() for method in entry['baselines'])
        if not complete:
            records.append(dict(dataset_id=entry['dataset_id'],status='incomplete_training'))
            continue
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        def limits():
            os.sched_setaffinity(0,{study['resources']['cpus'][0]})
            cap=int(study['resources']['memory_gib']*1024**3)
            resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
        with (logs/(entry['dataset_id']+'_analysis.log')).open('x') as log:
            code=subprocess.run([sys.executable,str(endpoint/'source/analyze_endpoint.py'),
                '--root',str(endpoint)],stdout=log,stderr=subprocess.STDOUT,env=env,preexec_fn=limits).returncode
        record=dict(dataset_id=entry['dataset_id'],returncode=code,
                    status='independently_verified' if code==0 else 'verification_failed')
        records.append(record)
        print(json.dumps(record),flush=True)
        dispatch.save(root/'analysis_status.json',dict(at=dispatch.now(),study_sha256=digest,records=records))
    success=result.returncode==0 and len(records)==len(study['datasets']) and all(
        r['status']=='independently_verified' for r in records)
    dispatch.save(root/'completion.json',dict(at=dispatch.now(),study_sha256=digest,
        status='independently_verified' if success else 'incomplete_or_failed',records=records))
    if success:
        with (logs/'aggregate_repeats.log').open('x') as log:
            code=subprocess.run([sys.executable,str(root/'source/aggregate_repeats.py'),'--root',str(root)],
                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')).returncode
        success=code==0
        dispatch.save(root/'aggregate_completion.json',dict(at=dispatch.now(),returncode=code,
            status='complete' if success else 'aggregation_failed'))
    return 0 if success else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    sys.exit(main(parser.parse_args().root))
