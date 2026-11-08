"""Validation-only learning-rate search (development cohort only).

plan_tuning returns the candidate suites and their jobs, so several searches
can share one process pool; finish_tuning finalizes and records selections.
"""
from __future__ import annotations
import copy
import numpy as np
from . import experiment as _exp
from .io import write_json,object_hash


DEFAULT_CANDIDATES=[{'lr_rec':.0003,'lr_out':.01},{'lr_rec':.001,'lr_out':.03},{'lr_rec':.003,'lr_out':.1}]


def plan_tuning(cfg,output,root,resume=False,serial=False):
    if cfg.stage!='development' or cfg.cell_cohort!='development':
        raise ValueError('Tuning is allowed only on the development cohort/stage')
    # Identical number of candidates, seeds, updates and examples per learner.
    # Candidates are fixed in advance; no test/analysis/intervention branch runs.
    candidates=cfg.tuning_candidates or DEFAULT_CANDIDATES
    representatives={}
    for c in cfg.conditions:
        if c.identity=='authentic' and c.topology=='type_prior' and c.training_noise_std==0:
            if c.learner in representatives:
                raise ValueError(f'Ambiguous tuning representative for learner {c.learner}; provide one intact/noiseless condition')
            representatives[c.learner]=c
    if not representatives:
        raise ValueError('Need at least one intact/type-prior/noiseless development condition for tuning')
    suites=[];all_jobs=[]
    for learner,condition in representatives.items():
        seen=set()
        for index,rates in enumerate(candidates):
            # A frozen reservoir ignores lr_rec: evaluate each distinct readout rate once.
            key=rates['lr_out'] if learner=='readout_only' else (rates['lr_rec'],rates['lr_out'])
            if key in seen: continue
            seen.add(key)
            run_cfg=copy.deepcopy(cfg);run_cfg.name=f'{cfg.name}_tune_{learner}_{index}'
            run_cfg.conditions=[condition];run_cfg.learning_rates={learner:rates}
            out=output/learner/f'candidate_{index}'
            if serial:
                # Serial path keeps the v7 contract: one run_suite call per candidate.
                runs=_exp.run_suite(run_cfg,out,project_root=root,resume=resume,test=False)
                suites.append((learner,index,rates,run_cfg,out,runs,None));continue
            jobs=_exp.prepare_suite(run_cfg,out,project_root=root,resume=resume,test=False)
            suites.append((learner,index,rates,run_cfg,out,len(all_jobs),len(jobs)));all_jobs.extend(jobs)
    return suites,all_jobs


def finish_tuning(cfg,output,suites,results,root):
    candidates=cfg.tuning_candidates or DEFAULT_CANDIDATES
    summary=[];selected={};by_learner={}
    for learner,index,rates,run_cfg,out,offset,n in suites:
        runs=offset if n is None else _exp.finalize_suite(run_cfg,out,results[offset:offset+n],project_root=root,test=False)
        scores=np.asarray([r['best_validation_bce'] for r in runs],dtype=float)
        if len(scores)!=len(cfg.seeds) or not np.all(np.isfinite(scores)):
            raise ValueError('Tuning candidate has missing or nonfinite validation results')
        entry={'learner':learner,'candidate':index,**rates,'mean_validation_bce':float(scores.mean()),
               'mean_validation_accuracy':float(np.mean([r.get('validation_accuracy',np.nan) for r in runs])),
               'mean_best_step':float(np.mean([r.get('best_step',np.nan) for r in runs])),
               'n_seeds':len(runs),'updates_per_run':cfg.training_steps,'test_evaluated':False}
        summary.append(entry);by_learner.setdefault(learner,[]).append(entry)
    for learner,rows in by_learner.items():
        best=min(rows,key=lambda r:(r['mean_validation_bce'],r['candidate']))
        selected[learner]={k:best[k] for k in ['lr_rec','lr_out']}
    result=cfg.as_dict();result['learning_rates']={**cfg.learning_rates,**selected}
    write_json(output/'tuning_summary.json',summary)
    write_json(output/'selected_development_config.json',result)
    write_json(output/'selection_record.json',{'criterion':'mean per-seed best validation BCE',
                'candidates_per_learner':len(candidates),'candidates':candidates,'test_evaluated':False,
                'source_config_hash':object_hash(cfg.as_dict()),'selected_learning_rates':selected,
                'scope':'Noisy and perturbation conditions require their own matched-noise tuning study.'})
    return selected
