"""Strict serializable experiment configuration and factorial condition plans."""
from __future__ import annotations
from dataclasses import dataclass,field,asdict,fields
from pathlib import Path
import json
import numpy as np
from .task import TaskConfig


@dataclass(frozen=True)
class Condition:
    name: str
    learner: str = 'bptt'
    identity: str = 'authentic'
    topology: str = 'type_prior'
    feedback_mode: str = 'random'
    training_noise_std: float = 0.
    calibration: str = 'matched'
    # v11 Family D: per-condition operating point (fraction of each cell's ceiling).
    baseline_fraction: float | None = None

    def __post_init__(self):
        if self.baseline_fraction is not None and not 0 <= self.baseline_fraction < 1:
            raise ValueError('baseline_fraction must lie in [0, 1)')
        if self.calibration not in {'matched','reference'}:
            raise ValueError("calibration must be 'matched' or 'reference'")
        if not self.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.name):
            raise ValueError('Condition names must be nonempty filesystem-safe identifiers')
        if self.learner not in {'bptt','local','local_no_beta','readout_only','node_perturbation'}:
            raise ValueError(f'Unknown learner {self.learner}')
        if self.identity not in {'authentic','within_type_shuffle','type_homogeneous','fast_decay','cross_type_permutation'}:
            raise ValueError(f'Unknown identity {self.identity}')
        if self.topology not in {'type_prior','degree_rewired','edge_count_random'}:
            raise ValueError(f'Unknown topology {self.topology}')
        if self.feedback_mode not in {'random','symmetric'}:
            raise ValueError('feedback_mode must be random or symmetric')
        if not np.isfinite(self.training_noise_std) or self.training_noise_std < 0:
            raise ValueError('training_noise_std must be nonnegative')
        if self.learner == 'node_perturbation' and self.training_noise_std <= 0:
            raise ValueError('Node perturbation requires an explicit positive training_noise_std')


def condition_plan(suite='core'):
    core = [Condition('intact_bptt'),Condition('intact_local','local'),
            Condition('intact_no_beta','local_no_beta'),Condition('readout_only','readout_only')]
    if suite == 'core':
        return core
    factorial = [Condition(f'{identity}__{topology}__{learner}',learner,identity,topology)
                 for identity in ('authentic','within_type_shuffle')
                 for topology in ('type_prior','degree_rewired')
                 for learner in ('bptt','local')]
    if suite == 'factorial':
        return factorial
    if suite == 'full':
        return factorial + [
            Condition('local_no_beta','local_no_beta'),Condition('readout_only','readout_only'),
            Condition('type_homogeneous_local','local','type_homogeneous'),
            Condition('fast_decay_local','local','fast_decay'),
            Condition('cross_type_local','local','cross_type_permutation'),
            Condition('node_perturbation','node_perturbation',training_noise_std=.1),
            Condition('bptt_noise_matched','bptt',training_noise_std=.1),
        ]
    raise ValueError('suite must be core, factorial, or full')


@dataclass
class ExperimentConfig:
    name: str = 'development_pilot'
    stage: str = 'development'
    cell_cohort: str = 'development'
    seeds: tuple[int,...] = (42,101,202)
    cell_counts: tuple[int,...] = (24,4,2,2)
    task: TaskConfig = field(default_factory=TaskConfig)
    conditions: list[Condition] = field(default_factory=condition_plan)
    training_steps: int = 300
    batch_size: int = 32
    validation_interval: int = 25
    validation_trials: int = 256
    test_trials: int = 512
    analysis_trials: int = 128
    checkpoint_steps: tuple[int,...] = (0,100,300)
    optimizer: str = 'adam'
    lr_rec: float = .003
    lr_out: float = .03
    learning_rates: dict = field(default_factory=dict)
    tuning_candidates: list = field(default_factory=list)
    max_grad_norm: float | None = 1.
    max_weight: float | None = 2.
    recurrent_gain: float = .6
    inhibitory_gain: float = 1.5
    input_gain: float = 1.
    input_mode: str = 'distributed'
    reference_rate_hz: float = 100.
    baseline_fraction: float = .05
    input_scale_fraction: float = .25
    min_validation_evr: float = .5
    cohort_unit: str = 'specimen'
    holdout_fraction: float = .25
    split_seed: int = 1701
    ood_delays: tuple[int,...] = (20,30)
    ood_noise_std: float = .5
    ood_distractor_std: float = .75
    evaluation_noise_std: float = 0.
    intervention_doses: tuple[float,...] = (0.5,1.)
    run_interventions: bool = True
    run_psychometrics: bool = True
    save_trajectories: bool = True
    run_fixed_points: bool = False

    def validate(self):
        if self.stage not in {'development','confirmation'} or self.cell_cohort not in {'all','development','confirmation'}:
            raise ValueError('Invalid stage or cell cohort')
        if self.stage == 'confirmation' and self.cell_cohort != 'confirmation':
            raise ValueError('Confirmation stage requires the reserved specimen cohort')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(int(s)!=s or s<0 for s in self.seeds):
            raise ValueError('seeds must be unique nonnegative integers')
        if len(self.cell_counts)!=4 or any(int(n)!=n or n<1 for n in self.cell_counts):
            raise ValueError('cell_counts must contain four positive integer counts')
        if self.task.bin_size_ms != 20:
            raise ValueError('Bundled parameters are fitted at 20 ms; do not rescale dt')
        for name in ['training_steps','batch_size','validation_interval','validation_trials','test_trials','analysis_trials']:
            v=getattr(self,name)
            if int(v)!=v or v<1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ['batch_size','validation_trials','test_trials','analysis_trials']:
            if getattr(self,name)%8:
                raise ValueError(f'{name} must be divisible by 8 for exact factorial trial balance')
        if not self.conditions or len({c.name for c in self.conditions}) != len(self.conditions):
            raise ValueError('Conditions must have unique names')
        if any(d in self.task.delay_bins for d in self.ood_delays) or any(int(d)!=d or d<=0 for d in self.ood_delays):
            raise ValueError('OOD delay support must be positive and disjoint from training')
        if not self.ood_delays:
            raise ValueError('Provide at least one OOD delay')
        if self.ood_noise_std <= self.task.noise_std:
            raise ValueError('OOD noise must exceed training sensory noise')
        if self.ood_distractor_std <= self.task.distractor_std:
            raise ValueError('OOD distractor strength must exceed training distractor strength')
        for name in ['lr_rec','lr_out','recurrent_gain','inhibitory_gain','input_gain',
                     'baseline_fraction','input_scale_fraction','min_validation_evr',
                     'ood_noise_std','ood_distractor_std','evaluation_noise_std','reference_rate_hz']:
            v=getattr(self,name)
            if not np.isfinite(v) or v<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.optimizer not in {'adam','sgd','rmsprop'}:
            raise ValueError('optimizer must be adam, sgd or rmsprop')
        if self.cohort_unit not in {'specimen','donor'} or not 0<self.holdout_fraction<1:
            raise ValueError('Invalid cohort split')
        if self.input_mode not in {'distributed','vip_context'}:
            raise ValueError('Invalid input_mode')
        if not 0 <= self.baseline_fraction < 1 or self.input_scale_fraction<=0 or self.reference_rate_hz<=0:
            raise ValueError('Invalid current/rate normalization')
        if not self.intervention_doses or any(not 0<=d<=1 for d in self.intervention_doses):
            raise ValueError('Intervention doses must lie in [0,1]')
        if any(int(v)!=v or v<0 for v in self.checkpoint_steps):
            raise ValueError('checkpoint_steps must be nonnegative integers')
        if any(v>self.training_steps for v in self.checkpoint_steps):
            raise ValueError('checkpoint_steps cannot exceed training_steps')
        for name in ['max_grad_norm','max_weight']:
            v=getattr(self,name)
            if v is not None and (not np.isfinite(v) or v<=0):
                raise ValueError(f'{name} must be positive or null')
        valid_learners = {c.learner for c in self.conditions}
        for learner, rates in self.learning_rates.items():
            if learner not in valid_learners or set(rates) != {'lr_rec','lr_out'} or any(not np.isfinite(v) or v<0 for v in rates.values()):
                raise ValueError('learning_rates must map configured learners to nonnegative lr_rec/lr_out pairs')
        return self

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls,raw):
        raw=dict(raw)
        unknown=set(raw)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown experiment fields: {sorted(unknown)}')
        if 'task' in raw:
            raw['task']=TaskConfig(**raw['task'])
        if 'conditions' in raw:
            raw['conditions']=[Condition(**c) for c in raw['conditions']]
        for name in ['seeds','cell_counts','checkpoint_steps','ood_delays','intervention_doses']:
            if name in raw:
                raw[name]=tuple(raw[name])
        return cls(**raw).validate()

    @classmethod
    def load(cls,path):
        return cls.from_dict(json.loads(Path(path).read_text()))
