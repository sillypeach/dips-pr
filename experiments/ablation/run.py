#!/usr/bin/env python3
"""Reproduce Table 4 or run selected frozen component paths with independent audits."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'pruning'))
from common import archived_points,saved_points,report,run_paths

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('paper','summarize','run','smoke'):
        q=sub.add_parser(action);q.add_argument('--output',required=True,type=Path)
        if action=='summarize':q.add_argument('--input',required=True,type=Path)
        if action=='run':
            q.add_argument('--inputs',type=Path,help='Shared synthetic inputs directory')
            g=q.add_mutually_exclusive_group(required=True);g.add_argument('--dataset',nargs='+');g.add_argument('--all',action='store_true',help='All 15 datasets and five arms: 75 full paths')
            q.add_argument('--arm',nargs='+',choices=['none','v_only','u_only','vu_single','vu_two'],help='Default: three displayed arms; --all uses all five')
    args=p.parse_args(argv)
    if args.action=='paper':cfg,rows=archived_points(ROOT);report(ROOT,cfg,rows,args.output)
    elif args.action=='summarize':cfg,rows=saved_points(ROOT,args.input);report(ROOT,cfg,rows,args.output)
    else:
        if args.action=='run' and args.all and args.arm:p.error('--all already selects all five arms; do not combine it with --arm')
        run_paths(ROOT,args,smoke=args.action=='smoke')

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as error:print(str(error),file=sys.stderr);sys.exit(1)
