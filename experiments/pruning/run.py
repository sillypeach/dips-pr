#!/usr/bin/env python3
"""Reproduce Table 3 from archives, count a dictionary, or fit frozen outer paths."""
from pathlib import Path
import argparse
import sys
from common import config, archived_points, saved_points, report, run_paths, kernels, load_dataset, new_output, write
ROOT=Path(__file__).resolve().parent

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    for action in ('paper','summarize','run','count','smoke'):
        q=sub.add_parser(action);q.add_argument('--output',required=True,type=Path)
        if action=='summarize':q.add_argument('--input',required=True,type=Path)
        if action in ('run','count'):
            q.add_argument('--inputs',type=Path,help='Shared synthetic inputs directory')
            if action=='run':
                g=q.add_mutually_exclusive_group(required=True);g.add_argument('--dataset',nargs='+');g.add_argument('--all',action='store_true')
            else:q.add_argument('--dataset',required=True)
    args=p.parse_args(argv)
    if args.action=='paper':cfg,rows=archived_points(ROOT);report(ROOT,cfg,rows,args.output)
    elif args.action=='summarize':cfg,rows=saved_points(ROOT,args.input);report(ROOT,cfg,rows,args.output)
    elif args.action in ('run','smoke'):run_paths(ROOT,args,smoke=args.action=='smoke')
    else:
        cfg=config(ROOT);out=new_output(args.output)
        entry,tx,y,fitcfg,proof=load_dataset(ROOT,cfg,args.dataset,args.inputs)
        counter=kernels(ROOT)[-1];result=counter.count_dictionary(tx,**proof['dictionary_config']);result['provenance']=proof
        write(out/'dictionary.json',result);print('Dictionary:',result['eligible_patterns'],result['status'])

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as error:print(str(error),file=sys.stderr);sys.exit(1)
