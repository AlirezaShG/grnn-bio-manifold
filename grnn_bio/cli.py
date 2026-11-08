"""Command-line research workflows. All runs are explicit, local and resumable."""
from __future__ import annotations
import argparse,json,subprocess,sys,copy
from pathlib import Path
import numpy as np
from .config import ExperimentConfig,condition_plan
from .io import write_json,object_hash


def main(argv=None):
    parser=argparse.ArgumentParser(prog='grnn-bio',description='Audited fixed-physiology GFR research experiments')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['run','plan','tune']:
        p=commands.add_parser(name)
        p.add_argument('--config',required=True,type=Path)
        p.add_argument('--output',required=name!='plan',type=Path)
        p.add_argument('--resume',action='store_true')
        p.add_argument('--jobs',type=int,default=1,help='parallel worker processes')
    p=commands.add_parser('validate');p.add_argument('--output',type=Path,default=Path('results/validation/validation.json'))
    p=commands.add_parser('plot');p.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parent.parent
    if args.command=='validate':
        from .validation import validate_release
        report=validate_release(root,args.output)
        print(json.dumps(report,indent=2));return 0 if report['status']=='pass' else 1
    if args.command=='plot':
        from .plotting import plot_results
        plot_results(args.output);return 0
    cfg=ExperimentConfig.load(args.config)
    if args.command=='plan':
        payload={'name':cfg.name,'stage':cfg.stage,'conditions':[c.name for c in cfg.conditions],
                 'n_runs':len(cfg.seeds)*len(cfg.conditions),'network_seeds':cfg.seeds,
                 'neurons':sum(cfg.cell_counts),'updates_per_run':cfg.training_steps,
                 'training_trials_per_run':cfg.training_steps*cfg.batch_size,
                 'train_delay_ms':[d*cfg.task.bin_size_ms for d in cfg.task.delay_bins],
                 'ood_delay_ms':[d*cfg.task.bin_size_ms for d in cfg.ood_delays],
                 'selected_checkpoint':'lowest validation BCE; all updates still performed',
                 'primary_factorial_metric':'test_delay / accuracy_incongruent',
                 'confirmation_requires':'locked development-selected settings, complete paired runs, independent reserved cohort'}
        print(json.dumps(payload,indent=2));
        if args.output: write_json(args.output,payload)
        return 0
    from .experiment import run_suite,prepare_suite,execute_jobs,finalize_suite
    if args.command=='run':
        run_suite(cfg,args.output,project_root=root,resume=args.resume,jobs=args.jobs);return 0
    from .tuning import plan_tuning,finish_tuning
    suites,all_jobs=plan_tuning(cfg,args.output,root,args.resume,serial=args.jobs<=1)
    results=execute_jobs(all_jobs,args.jobs) if all_jobs else []
    finish_tuning(cfg,args.output,suites,results,root)
    print(f'Saved validation-only selections to {args.output}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
