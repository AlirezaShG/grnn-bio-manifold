"""Reproducible paired training, held-out evaluation and post-training analysis."""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import json,time,copy
import numpy as np
import pandas as pd
from .config import ExperimentConfig,Condition
from .data import load_dataset,partition_cells,sample_cells
from .model import calibrate_current
from .circuit import Circuit
from .randomness import rng_for,seed_for
from .task import generate_batch,behavior_metrics,decision_values
from .learners import (exact_bptt_gradients,local_eligibility_gradients,readout_only_gradients,
                       node_perturbation_gradients,ProjectedOptimizer)
from .analysis import fit_population_analysis,phase_average,temporal_generalization
from .statistics import mean_interval,paired_condition_table,network_level_cell_association
from .interventions import evaluate_rate_interventions,evaluate_pulse_recovery
from .io import (write_json,save_npz,sha256_file,object_hash,source_fingerprint,environment,
                 save_checkpoint,load_checkpoint)


def _cosine(a,b):
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denom) if denom>1e-18 else None


def gradient_alignment(circuit,batch,exact=None):
    """Cosine of each local estimate with the exact recurrent gradient.

    'symmetric' feedback isolates the error from cutting paths through other
    neurons; 'random' adds the error from the fixed feedback vector.
    """
    exact=exact_bptt_gradients(circuit,batch) if exact is None else exact
    out={'full_gradient_norm':float(np.linalg.norm(exact['rec'])),
         'readout_feedback_cosine':_cosine(circuit.W_out,circuit.feedback)}
    for beta in (True,False):
        for mode in ('random','symmetric'):
            est=local_eligibility_gradients(circuit,batch,include_beta=beta,feedback_mode=mode)
            key=('local' if beta else 'no_beta')+'_'+mode
            out[key+'_cosine']=_cosine(est['rec'],exact['rec'])
            out[key+'_norm']=float(np.linalg.norm(est['rec']))
    return out


def build_circuit(cfg,df,condition,seed):
    params,rows=sample_cells(df,cfg.cell_counts,seed_for(seed,'cell_sample'))
    frac=cfg.baseline_fraction if condition.baseline_fraction is None else condition.baseline_fraction
    calibration=calibrate_current(params,frac,cfg.input_scale_fraction)
    circuit=Circuit(params,rows.cell_type.to_numpy(),seed=seed_for(seed,'weights'),
                    identity=condition.identity,topology=condition.topology,
                    recurrent_gain=cfg.recurrent_gain,inhibitory_gain=cfg.inhibitory_gain,
                    input_gain=cfg.input_gain,input_mode=cfg.input_mode,
                    calibration=calibration,reference_rate_hz=cfg.reference_rate_hz,
                    calibration_mode=condition.calibration,baseline_fraction=frac,
                    input_scale_fraction=cfg.input_scale_fraction)
    return circuit,rows


def evaluate(circuit,batch,current_noise=None):
    traj=circuit.forward(batch.X,current_noise=current_noise,record_currents=True)
    metrics=behavior_metrics(traj.logits,batch)
    valid=np.asarray(batch.meta['valid_choice'])
    weights=batch.decision_mask/batch.decision_mask.sum(axis=1,keepdims=True)
    per_trial=(np.logaddexp(0.,-batch.y[:,None]*traj.logits)*weights).sum(1)
    metrics['bce']=float(per_trial[valid].mean()) if np.any(valid) else float('nan')
    vm=batch.valid_mask
    rates=traj.rates_hz[vm]
    ceilings=np.broadcast_to(circuit.gamma_hz,traj.rates_hz.shape)[vm]
    metrics.update({'mean_rate_hz':float(rates.mean()),'silent_rate_fraction':float(np.mean(rates<=1e-12)),
                    'ceiling_rate_fraction':float(np.mean(rates>=.95*ceilings)),
                    'mean_abs_current_pa':float(np.abs(traj.currents_pa[vm]).mean()),
                    'current_above_fit_scale_fraction':float(np.mean(np.abs(traj.currents_pa[vm])>circuit.params.max_current))})
    return metrics,traj,per_trial


def noise_for(cfg,seed,namespace,batch,n):
    if cfg.evaluation_noise_std == 0:
        return None
    return rng_for(seed,namespace).normal(0.,cfg.evaluation_noise_std,(batch.n_trials,batch.n_steps,n))


def _test_batches(cfg,seed):
    settings=[('test_id',{}),('test_delay',{'delay_bins':cfg.ood_delays}),
              ('test_sensory_noise',{'noise_std':cfg.ood_noise_std}),
              ('test_distractor',{'distractor_std':cfg.ood_distractor_std}),
              ('test_combined',{'delay_bins':cfg.ood_delays,'noise_std':cfg.ood_noise_std,'distractor_std':cfg.ood_distractor_std})]
    for name,kwargs in settings:
        yield name,generate_batch(cfg.task,cfg.test_trials,rng_for(seed,cfg.stage+'/'+name),**kwargs)


def _gradient_step(circuit,batch,condition,cfg,seed,step):
    noise_rng=rng_for(seed,f'{cfg.stage}/train_noise/{step}')
    if condition.learner=='node_perturbation':
        return node_perturbation_gradients(circuit,batch,rng=noise_rng,sigma=condition.training_noise_std)
    noise=None
    if condition.training_noise_std:
        # Time-major drawing exactly matches the node-perturbation stream.
        noise=np.stack([noise_rng.normal(0.,condition.training_noise_std,(batch.n_trials,circuit.N)) for _ in range(batch.n_steps)],axis=1)
    if condition.learner=='bptt': return exact_bptt_gradients(circuit,batch,noise=noise)
    if condition.learner=='readout_only': return readout_only_gradients(circuit,batch,noise=noise)
    return local_eligibility_gradients(circuit,batch,noise=noise,include_beta=condition.learner!='local_no_beta',feedback_mode=condition.feedback_mode)


def run_one(cfg,df,condition,seed,run_dir,*,project_root,resume=False,test=True):
    """Train for a fixed budget; choose only by validation BCE; test selected model.

    Data draws are indexed by (stage, seed, namespace, training step), so adding
    validation, a new condition, or resuming cannot change any training example.
    Test=False never calls the test generator and is used by the tuning command.
    """
    start_time=time.perf_counter()
    run_dir=Path(run_dir)
    spec={'experiment':cfg.as_dict(),'condition':asdict(condition),'seed':int(seed),
          'source_hash':source_fingerprint(project_root),'test_enabled':bool(test),
          'input_hashes':{name:sha256_file(Path(project_root)/name) for name in
                          ['model/gfr_dataset.json','data/metadata.csv']},
          'cohort_hash':object_hash(df[['cell_id','cell_type','params']].sort_values('cell_id').to_dict('records'))}
    cfg_hash=object_hash(spec)
    if run_dir.exists() and any(run_dir.iterdir()):
        if not resume:
            raise FileExistsError(f'{run_dir} is nonempty. Choose another output directory or use --resume.')
        manifest=json.loads((run_dir/'manifest.json').read_text())
        if manifest['config_hash']!=cfg_hash:
            raise ValueError('Existing run has different code or config; resume refused')
        if (run_dir/'completed.json').exists():
            return json.loads((run_dir/'completed.json').read_text())
    run_dir.mkdir(parents=True,exist_ok=True)
    circuit,rows=build_circuit(cfg,df,condition,seed)
    learning_rates=cfg.learning_rates.get(condition.learner,{'lr_rec':cfg.lr_rec,'lr_out':cfg.lr_out})
    optimizer=ProjectedOptimizer(kind=cfg.optimizer,**learning_rates,max_grad_norm=cfg.max_grad_norm,max_weight=cfg.max_weight)
    val=generate_batch(cfg.task,cfg.validation_trials,rng_for(seed,cfg.stage+'/validation'))
    val_noise=noise_for(cfg,seed,cfg.stage+'/validation_noise',val,circuit.N)
    write_json(run_dir/'manifest.json',{'config_hash':cfg_hash,**spec,'environment':environment(),
               'cell_seed':seed_for(seed,'cell_sample'),'weight_seed':seed_for(seed,'weights'),
               'calibration':circuit.calibration_info,'calibration_mode':circuit.calibration_mode,
               'reference_calibration':circuit.reference_calibration_info,'identity_provenance':circuit.identity_provenance,
               'topology_diagnostics':circuit.rewire_info,'sampled_cell_count':len(rows),
               'unique_cell_count':int(rows.cell_id.nunique()),'n_synapses':int(circuit.mask.sum()),
               'plastic_recurrent_parameters':0 if condition.learner=='readout_only' else int(circuit.mask.sum()),
               'readout_parameters':int(circuit.readout_mask.sum()+1),
               'checkpoint_selection':'minimum validation decision-bin BCE',
               'neuron_parameters_trainable':False,'no_measured_connectome':True,
               'confirmation_claim':False if cfg.stage=='development' else 'requires locked protocol and completed design'})
    keep=['slot_index','cell_id','cell_type','cre-line','structure__layer','tag__dendrite_type','val_evr','test_evr','donor__id']
    rows[[x for x in keep if x in rows]].to_csv(run_dir/'sampled_cells.csv',index=False)
    start_step=0;history=[];total_bins=0;training_seconds=0.
    best_val,_,_=evaluate(circuit,val,val_noise)
    best_loss=best_val['bce'];best_step=0
    last_path=run_dir/'last_checkpoint.npz';best_path=run_dir/'best_checkpoint.npz'
    if resume and last_path.exists():
        restored=load_checkpoint(last_path,circuit,optimizer,cfg_hash)
        start_step=restored['step'];best_loss=restored['best_validation_bce'];best_step=restored['best_step']
        total_bins=restored['simulated_trial_bins'];training_seconds=restored.get('training_seconds',0.)
        history=json.loads((run_dir/'history.json').read_text())
        history=[x for x in history if x['step']<=start_step]
    else:
        init_meta={'config_hash':cfg_hash,'step':0,'best_step':0,'best_validation_bce':best_loss,'simulated_trial_bins':0,'training_seconds':0.}
        save_checkpoint(best_path,circuit,optimizer,init_meta)
        if 0 in cfg.checkpoint_steps: save_npz(run_dir/'checkpoints/step_000000.npz',**circuit.state_dict())
    for step in range(start_step+1,cfg.training_steps+1):
        batch=generate_batch(cfg.task,cfg.batch_size,rng_for(seed,f'{cfg.stage}/train/{step}'))
        t0=time.perf_counter()
        result=_gradient_step(circuit,batch,condition,cfg,seed,step)
        opt_info=optimizer.step(circuit,result)
        training_seconds+=time.perf_counter()-t0
        total_bins+=int(result['diagnostics']['simulated_trial_bins'])
        row={'step':step,'training_loss':result['loss'],'training_accuracy':result['accuracy'],
             'simulated_trial_bins':total_bins,'training_seconds':training_seconds,**opt_info}
        if step%cfg.validation_interval==0 or step==cfg.training_steps:
            circuit.assert_constraints()
            vm,_,_=evaluate(circuit,val,val_noise)
            row.update({'validation_'+k:v for k,v in vm.items()})
            improved=vm['bce']<best_loss
            if improved: best_loss=vm['bce'];best_step=step
            meta={'config_hash':cfg_hash,'step':step,'best_step':best_step,'best_validation_bce':best_loss,
                  'simulated_trial_bins':total_bins,'training_seconds':training_seconds}
            if improved: save_checkpoint(best_path,circuit,optimizer,meta)
            history.append(row)
            write_json(run_dir/'history.json',history)
            save_checkpoint(last_path,circuit,optimizer,meta)
        else:
            history.append(row)
        if step in cfg.checkpoint_steps:
            save_npz(run_dir/f'checkpoints/step_{step:06d}.npz',**circuit.state_dict())
    pd.DataFrame(history).to_csv(run_dir/'learning_curve.csv',index=False)
    load_checkpoint(best_path,circuit,optimizer,cfg_hash)
    circuit.assert_constraints()
    final_val,_,_=evaluate(circuit,val,val_noise)
    summary={'condition':condition.name,'seed':int(seed),'identity':condition.identity,
             'topology':condition.topology,'learner':condition.learner,'best_step':best_step,
             'best_validation_bce':final_val['bce'],'validation_accuracy':final_val['accuracy'],
             'trained_steps':cfg.training_steps,'training_trials':cfg.training_steps*cfg.batch_size,
             'simulated_trial_bins':total_bins,'training_seconds':training_seconds,'test_enabled':test,
             'recurrent_weight_change_l2':float(np.linalg.norm(circuit.W_abs-circuit.W_abs_initial)),
             'metrics':[]}
    if test:
        all_trial_rows=[]
        id_batch=id_traj=id_noise=None
        for split,batch in _test_batches(cfg,seed):
            noise=noise_for(cfg,seed,cfg.stage+'/'+split+'/noise',batch,circuit.N)
            metrics,traj,losses=evaluate(circuit,batch,noise)
            summary['metrics'].append({'split':split,'condition':condition.name,'seed':int(seed),
                                       'identity':condition.identity,'topology':condition.topology,'learner':condition.learner,
                                       'training_noise_std':condition.training_noise_std,**metrics})
            values=decision_values(traj.logits,batch)
            for i in range(batch.n_trials):
                all_trial_rows.append({'condition':condition.name,'seed':int(seed),'split':split,'trial_id':i,
                    'target':float(batch.y[i]),'logit':float(values[i]),'correct':bool((1 if values[i]>=0 else -1)==batch.y[i]),
                    'bce':float(losses[i]),'context':int(batch.meta['context_rule'][i]),
                    'congruent':bool(batch.meta['is_congruent'][i]),'delay_bins':int(batch.meta['delay_bins'][i]),
                    'relevant_coherence':float(batch.meta['relevant_coherence'][i]),'irrelevant_coherence':float(batch.meta['irrelevant_coherence'][i])})
            if split=='test_id': id_batch,id_traj,id_noise=batch,traj,noise
        pd.DataFrame(all_trial_rows).to_csv(run_dir/'trial_predictions.csv',index=False)
        # Shortcut controls reuse precisely the ID trials and exogenous noise.
        shortcut_rows=[]
        for control in ['context_removed','recurrent_removed']:
            if control=='context_removed':
                changed=copy.deepcopy(id_batch); changed.X[:,:,2:4]=0.
                metric,_,_=evaluate(circuit,changed,id_noise)
            else:
                old=circuit.W_abs.copy();circuit.W_abs[:]=0.
                try: metric,_,_=evaluate(circuit,id_batch,id_noise)
                finally: circuit.W_abs[:]=old
            shortcut_rows.append({'control':control,**metric})
        write_json(run_dir/'shortcut_controls.json',shortcut_rows)
        analysis_fit=generate_batch(cfg.task,cfg.analysis_trials,rng_for(seed,cfg.stage+'/analysis_fit'))
        analysis_eval=generate_batch(cfg.task,cfg.analysis_trials,rng_for(seed,cfg.stage+'/analysis_eval'))
        fit_noise=noise_for(cfg,seed,cfg.stage+'/analysis_fit_noise',analysis_fit,circuit.N)
        eval_noise=noise_for(cfg,seed,cfg.stage+'/analysis_eval_noise',analysis_eval,circuit.N)
        tf=circuit.forward(analysis_fit.X,current_noise=fit_noise)
        te=circuit.forward(analysis_eval.X,current_noise=eval_noise)
        diagnostic_batch=generate_batch(cfg.task,min(32,cfg.analysis_trials),rng_for(seed,cfg.stage+'/gradient_diagnostic'))
        exact=exact_bptt_gradients(circuit,diagnostic_batch)
        write_json(run_dir/'gradient_diagnostics.json',{**gradient_alignment(circuit,diagnostic_batch,exact),
            'temporal_full_gradient_rms':exact['diagnostics']['temporal_gradient_rms'],
            'interpretation':'approximation alignment on independent diagnostic trials; not a full-gradient identity'})
        analysis_rows=[]
        for phase in ['stimulus','delay','decision']:
            fitted=fit_population_analysis(tf.rates_norm,analysis_fit,phase=phase)
            analysis_rows.append(fitted.evaluate(te.rates_norm,analysis_eval,readout=circuit.W_out))
            if phase=='delay':
                write_json(run_dir/'pca_condition_trajectories.json',fitted.condition_trajectories(te.rates_norm,analysis_eval))
                save_npz(run_dir/'fitted_delay_analysis.npz',pca_mean=fitted.pca_mean,pca_components=fitted.pca_components,
                         choice_axis=fitted.choice_axis,context_axis=fitted.context_axis)
        write_json(run_dir/'population_analysis.json',analysis_rows)
        temporal=temporal_generalization(tf.rates_norm,analysis_fit,te.rates_norm,analysis_eval,
                                        n_bins=min(5,min(cfg.task.delay_bins)))
        write_json(run_dir/'temporal_generalization.json',temporal)
        features=phase_average(te.rates_hz,analysis_eval,'delay')
        pos=features[analysis_eval.y>0];neg=features[analysis_eval.y<0]
        pooled=np.sqrt((pos.var(0,ddof=1)+neg.var(0,ddof=1))/2)
        dprime=np.divide(pos.mean(0)-neg.mean(0),pooled,out=np.full(circuit.N,np.nan),where=pooled>1e-10)
        cell_rows=[]
        for i in range(circuit.N):
            cell_rows.append({'seed':int(seed),'slot':i,'cell_id':int(circuit.params.cell_ids[i]),'cell_type':str(circuit.cell_types[i]),
                'identity':condition.identity,'dc_input_coefficient':float(np.mean(circuit.a[i]/circuit.ds[i])),
                'dc_history_coefficient':float(np.mean(circuit.beta[i]/circuit.ds[i])),
                'delay_selectivity_dprime':float(dprime[i]),'mean_delay_rate_hz':float(features[:,i].mean()),
                'readout_weight':float(circuit.W_out[i]),'is_synthetic':condition.identity!='authentic',
                'in_degree':int(circuit.mask[i].sum()),'out_degree':int(circuit.mask[:,i].sum()),
                'ephys_tau_metadata':rows.iloc[i].get('ef__tau',np.nan),
                'ephys_adaptation_metadata':rows.iloc[i].get('ef__adaptation',np.nan),
                'ephys_fi_slope_metadata':rows.iloc[i].get('ef__f_i_curve_slope',np.nan),
                'metadata_applicable_to_intact_tuple':condition.identity=='authentic'})
        pd.DataFrame(cell_rows).to_csv(run_dir/'cell_role_descriptive.csv',index=False)
        if cfg.save_trajectories:
            save_npz(run_dir/'analysis_trajectories.npz',fit_rates_hz=tf.rates_hz,eval_rates_hz=te.rates_hz,
                     fit_y=analysis_fit.y,eval_y=analysis_eval.y,fit_delay_mask=analysis_fit.delay_mask,
                     eval_delay_mask=analysis_eval.delay_mask,fit_X=analysis_fit.X,eval_X=analysis_eval.X)
        # Same independent analysis-fit/eval bank at each checkpoint; no training dependence.
        ckpt_rows=[];selected_state=circuit.state_dict()
        for cp in sorted((run_dir/'checkpoints').glob('step_*.npz')):
            with np.load(cp,allow_pickle=False) as data: circuit.load_state_dict(data)
            cf=circuit.forward(analysis_fit.X,current_noise=fit_noise)
            ce=circuit.forward(analysis_eval.X,current_noise=eval_noise)
            fit=fit_population_analysis(cf.rates_norm,analysis_fit,phase='delay')
            ckpt_rows.append({'step':int(cp.stem.split('_')[1]),**fit.evaluate(ce.rates_norm,analysis_eval,circuit.W_out),
                              **{'grad_'+k:v for k,v in gradient_alignment(circuit,diagnostic_batch).items()}})
        circuit.load_state_dict(selected_state)
        write_json(run_dir/'checkpoint_geometry.json',ckpt_rows)
        if cfg.run_interventions:
            # Cap here only on an independent bank configured by analysis_trials;
            # never select intervention trials by baseline correctness.
            iv_rows,iv_trials=evaluate_rate_interventions(circuit,analysis_eval,doses=cfg.intervention_doses,current_noise=eval_noise,
                                                        control_seed=seed_for(seed,'intervention_control'))
            pd.DataFrame(iv_rows).to_csv(run_dir/'rate_interventions.csv',index=False)
            pd.DataFrame(iv_trials).to_csv(run_dir/'rate_intervention_trials.csv',index=False)
            pulse=(evaluate_pulse_recovery(circuit,analysis_eval,current_noise=eval_noise,pulse_seed=seed_for(seed,'pulse'))
                   if min(cfg.task.delay_bins)>=2 else
                   {'status':'not_applicable','reason':'at least two silent-delay bins are required to measure recovery'})
            write_json(run_dir/'pulse_recovery.json',pulse)
        if cfg.run_psychometrics:
            psych=[]
            for context in (0,1):
                for coh in (-1.,-.5,-.25,0.,.25,.5,1.):
                    batch=generate_batch(cfg.task,cfg.analysis_trials,rng_for(seed,f'{cfg.stage}/psych/{context}/{coh}'),relevant_coherence=coh,context_rule=context)
                    pnoise=noise_for(cfg,seed,f'{cfg.stage}/psych_noise/{context}/{coh}',batch,circuit.N)
                    m,_,_=evaluate(circuit,batch,pnoise)
                    psych.append({'context':context,'coherence':coh,**m})
            pd.DataFrame(psych).to_csv(run_dir/'psychometrics.csv',index=False)
        if cfg.run_fixed_points:
            from .dynamics import find_fixed_point
            h=circuit.forward(analysis_eval.X,record_states=True).states[0,-1]
            write_json(run_dir/'fixed_point.json',find_fixed_point(circuit,h))
    summary['wall_seconds']=time.perf_counter()-start_time
    write_json(run_dir/'completed.json',summary)
    return summary


def _run_job(job):
    """Process-pool entry point; BLAS thread limits come from the environment."""
    cfg,cohort,condition,seed,run_dir,root,resume,test=job
    return run_one(cfg,cohort,condition,seed,run_dir,project_root=root,resume=resume,test=test)


def execute_jobs(jobs,n_workers=1):
    """Run (cfg,cohort,condition,seed,run_dir,root,resume,test) jobs; keep input order.

    Every run is deterministic given its own named RNG streams, so results do
    not depend on worker count or completion order. Workers are spawned with
    single-threaded BLAS to avoid oversubscription.
    """
    jobs=list(jobs)
    if n_workers<=1 or len(jobs)<=1:
        out=[]
        for job in jobs:
            print(f'Running {job[2].name}, network seed {job[3]}',flush=True)
            out.append(_run_job(job))
        return out
    import os
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    for var in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[var]='1'
    results=[None]*len(jobs)
    with ProcessPoolExecutor(max_workers=int(n_workers),mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(_run_job,job):i for i,job in enumerate(jobs)}
        done=0
        for fut in as_completed(futures):
            i=futures[fut];results[i]=fut.result();done+=1
            r=results[i]
            print(f'[{done}/{len(jobs)}] {r["condition"]} seed {r["seed"]}: step {r["best_step"]}, '
                  f'val BCE {r["best_validation_bce"]:.4f}, wall {r["wall_seconds"]:.0f}s',flush=True)
    return results


def prepare_suite(cfg,output_dir,*,project_root=None,resume=False,test=True):
    """Validate, write manifests and provenance, and return the job list."""
    cfg.validate();root=Path(project_root or Path(__file__).resolve().parent.parent)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    suite_spec={'config':cfg.as_dict(),'source_hash':source_fingerprint(root),'test_enabled':bool(test),
                'input_hashes':{name:sha256_file(root/name) for name in ['model/gfr_dataset.json','data/metadata.csv']}}
    suite_hash=object_hash(suite_spec)
    suite_manifest=out/'suite_manifest.json'
    if suite_manifest.exists() and json.loads(suite_manifest.read_text())['suite_hash']!=suite_hash:
        raise ValueError('Output suite code, inputs, or configuration changed; use a new output directory')
    cfg_path=out/'config.json'
    if cfg_path.exists() and object_hash(json.loads(cfg_path.read_text()))!=object_hash(cfg.as_dict()):
        raise ValueError('Output directory already belongs to a different configuration')
    write_json(suite_manifest,{'suite_hash':suite_hash,**suite_spec})
    write_json(cfg_path,cfg.as_dict())
    full=load_dataset(root/'model/gfr_dataset.json',root/'data/metadata.csv',cfg.min_validation_evr)
    split=dict(holdout_fraction=cfg.holdout_fraction,seed=cfg.split_seed,unit=cfg.cohort_unit)
    cohort=partition_cells(full,cfg.cell_cohort,**split)
    dev=partition_cells(full,'development',**split);conf=partition_cells(full,'confirmation',**split)
    write_json(out/'data_provenance.json',{'upstream_commit':'835a535f6831b88339c6a5c17180fdbc504f7448',
       'dataset_sha256':sha256_file(root/'model/gfr_dataset.json'),'metadata_sha256':sha256_file(root/'data/metadata.csv'),
       'pool_counts':full.cell_type.value_counts().to_dict(),'cohort_counts':cohort.cell_type.value_counts().to_dict(),
       'development_confirmation_cell_overlap':len(set(dev.cell_id)&set(conf.cell_id)),
       'development_confirmation_donor_overlap':len(set(dev['donor__id'])&set(conf['donor__id'])),
       'cohort_unit':cfg.cohort_unit,'holdout_fraction':cfg.holdout_fraction,'split_seed':cfg.split_seed,
       'development_counts':dev.cell_type.value_counts().to_dict(),'confirmation_counts':conf.cell_type.value_counts().to_dict(),
       'cohort_selection':'species,cre/morphology,original validation EVR; no test EVR cutoff',
       'inherited_fit_split_caveat':'Upstream Noise1 contributes to both fitting and validation; no independent patch-clamp revalidation is claimed.'})
    return [(cfg,cohort,condition,seed,out/'runs'/condition.name/str(seed),root,resume,test)
            for condition in cfg.conditions for seed in cfg.seeds]


def run_suite(cfg,output_dir,*,project_root=None,resume=False,test=True,jobs=1):
    job_list=prepare_suite(cfg,output_dir,project_root=project_root,resume=resume,test=test)
    completed=execute_jobs(job_list,jobs)
    return finalize_suite(cfg,output_dir,completed,project_root=project_root,test=test)


def finalize_suite(cfg,output_dir,completed,*,project_root=None,test=True):
    root=Path(project_root or Path(__file__).resolve().parent.parent);out=Path(output_dir)
    metrics=[row for r in completed for row in r['metrics']]
    pd.DataFrame([{k:v for k,v in r.items() if k!='metrics'} for r in completed]).to_csv(out/'training_summary.csv',index=False)
    if metrics:
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        grouped=[]
        for (condition,split),frame in pd.DataFrame(metrics).groupby(['condition','split']):
            for metric in ['accuracy','accuracy_incongruent','bce']:
                grouped.append({'condition':condition,'split':split,'metric':metric,**mean_interval(frame[metric].to_numpy(),seed=1701)})
        write_json(out/'network_summary.json',grouped)
        # Explicit primary endpoint; all condition-vs-reference comparisons are
        # one Holm family. Other endpoints are descriptive unless predeclared.
        reference=cfg.conditions[0].name
        primary=[r for r in metrics if r['split']=='test_delay']
        comparisons=paired_condition_table(primary,reference,'accuracy_incongruent',seed=1701)
        write_json(out/'paired_exploratory_comparisons.json',comparisons)
        from .design import factorial_contrasts
        # Only default-protocol conditions can be factorial cells: a different
        # operating point, calibration or feedback would duplicate a cell.
        factorial_names={c.name for c in cfg.conditions if c.baseline_fraction is None
                         and c.calibration=='matched' and c.feedback_mode=='random'}
        write_json(out/'factorial_contrasts.json',
                   factorial_contrasts([r for r in primary if r['condition'] in factorial_names]))
        cell_associations=[]
        for condition in cfg.conditions:
            if condition.identity!='authentic':
                continue
            cell_frames=[pd.read_csv(p) for p in (out/'runs'/condition.name).glob('*/cell_role_descriptive.csv')]
            if not cell_frames:
                continue
            cells=pd.concat(cell_frames,ignore_index=True)
            cells=cells[(cells.cell_type=='Pyr') & np.isfinite(cells.delay_selectivity_dprime)].copy()
            cells['absolute_delay_selectivity']=np.abs(cells.delay_selectivity_dprime)
            if len(cells):
                association=network_level_cell_association(cells.to_dict('records'),'dc_input_coefficient','absolute_delay_selectivity')
                cell_associations.append({'condition':condition.name,'x':'dc_input_coefficient',
                                          'y':'absolute held-out delay selectivity','status':'exploratory, not causal',**association})
        write_json(out/'cell_role_network_associations.json',cell_associations)
        from .plotting import plot_results
        plot_results(out)
    write_json(out/'suite_completed.json',{'n_runs':len(completed),'test_evaluated':test,
       'stage':cfg.stage,'statistical_unit':'network_run_conditional_on_fitted_cell_resource',
       'publication_claim':'none; interpret completed experiments with the research protocol',
       'source_hash':source_fingerprint(root),'environment':environment()})
    return completed
