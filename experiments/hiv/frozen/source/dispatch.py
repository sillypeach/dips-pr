"""Dispatch a frozen, grouped HIV benchmark on a dedicated CPU pool."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_study(root):
    study = read(root / 'study.json')
    registered = read(root / 'registered.json')
    digest = hashlib.sha256((root / 'study.json').read_bytes()).hexdigest()
    if registered['study_sha256'] != digest:
        raise ValueError('Study registration mismatch')
    for relative, expected in study['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed registered file: ' + relative)
    ids = [entry['dataset_id'] for entry in study['datasets']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate endpoint')
    cpus = study['resources']['cpus']
    if not cpus or len(cpus) != len(set(cpus)) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError('Invalid dedicated CPU allocation')
    return study, digest


def initial_jobs(study):
    jobs = []
    for entry in study['datasets']:
        did = entry['dataset_id']
        jobs.extend(dict(dataset_id=did, kind='dips', name=f'fold{i}') for i in range(3))
        jobs.extend(dict(dataset_id=did, kind='baseline', name=name) for name in entry['baselines'])
    return jobs


def schedule_final(study, finished, final_added):
    """Final models require all three successful CV workers, never a partial CV."""
    ready, blocked = [], []
    for entry in study['datasets']:
        did = entry['dataset_id']
        if did in final_added:
            continue
        folds = [j for j in finished if j['dataset_id'] == did and j['kind'] == 'dips'
                 and j['name'] in ('fold0', 'fold1', 'fold2')]
        if len(folds) != 3:
            continue
        final_added.add(did)
        identity = dict(dataset_id=did, kind='dips', name='final')
        if all(j.get('returncode') == 0 for j in folds):
            ready.append(identity)
        else:
            blocked.append(dict(identity, status='blocked_failed_cv', returncode=None, ended_at=now()))
    return ready, blocked


def run(root):
    root = Path(root).resolve()
    study, digest = verify_study(root)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'controller_started.json').exists():
            raise RuntimeError('Previous attempt exists; never restart all jobs automatically')
        cpus = study['resources']['cpus']
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
            env[name] = '1'
        save(root / 'controller_started.json', dict(at=now(), pid=os.getpid(), cpus=cpus,
             study_sha256=digest, python=sys.executable))
        pending = initial_jobs(study)
        active, finished, final_added = {}, [], set()
        endpoints = {e['dataset_id']: root / e['root'] for e in study['datasets']}
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        while pending or active or len(final_added) != len(endpoints):
            for pid, job in list(active.items()):
                code = job['proc'].poll()
                if code is None:
                    continue
                job['log'].close()
                record = {k: v for k, v in job.items() if k not in ('proc', 'log')}
                record.update(returncode=code, ended_at=now())
                finished.append(record)
                del active[pid]
                print(json.dumps(record), flush=True)
            ready, blocked = schedule_final(study, finished, final_added)
            pending = ready + pending
            finished.extend(blocked)
            used = {j['cpu'] for j in active.values()}
            for cpu in cpus:
                if cpu in used or not pending:
                    continue
                identity = pending.pop(0)
                did, kind, name = [identity[k] for k in ('dataset_id', 'kind', 'name')]
                endpoint = endpoints[did]
                filename = 'hiv_dips.py' if kind == 'dips' else 'hiv_baseline.py'
                flag = '--split' if kind == 'dips' else '--method'
                command = [sys.executable, '-u', str(endpoint / 'source' / filename),
                           '--root', str(endpoint), flag, name]
                if kind == 'dips':
                    command.extend(['--cpu', str(cpu), '--memory-gib', str(study['resources']['memory_gib'])])
                def limit(cpu=cpu):
                    os.sched_setaffinity(0, {cpu})
                    cap = int(study['resources']['memory_gib'] * 1024**3)
                    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
                log = (logs / f'{did}_{kind}_{name}.log').open('x')
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        preexec_fn=limit)
                active[proc.pid] = dict(identity, cpu=cpu, pid=proc.pid, started_at=now(),
                                        command=command, proc=proc, log=log)
                print(json.dumps({k: v for k, v in active[proc.pid].items()
                                  if k not in ('proc', 'log')}), flush=True)
            save(root / 'controller_status.json', dict(at=now(), study_sha256=digest,
                 active=[{k: v for k, v in j.items() if k not in ('proc', 'log')} for j in active.values()],
                 pending=pending, finished=finished, final_added=sorted(final_added)))
            if pending or active:
                time.sleep(1)
        expected = sum(4 + len(e['baselines']) for e in study['datasets'])
        success = len(finished) == expected and all(j.get('returncode') == 0 for j in finished)
        save(root / 'controller_completion.json', dict(at=now(), study_sha256=digest,
             status='processes_complete' if success else 'failed_or_blocked', expected_jobs=expected,
             finished=finished, note='Complete process receipts do not replace independent result verification'))
        return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    sys.exit(run(parser.parse_args().root))
