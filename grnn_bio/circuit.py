"""A type-prior recurrent circuit built from immutable fitted GFR neurons.

M[i,j] means j -> i. All outgoing recurrent signs follow the assigned slot
(E/I) identity. There is no measured connectome and no dendritic-compartment
claim. Current-based synapses leave the fitted single-cell equation intact.
"""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Real
import copy
import numpy as np
from .model import activation_and_grad, cell_step, calibrate_current
from .data import CELL_TYPES, manipulate_identity
from .randomness import rng_for

# Illustrative priors inspired by cortical inhibitory motifs. These numbers
# are modeling assumptions, NOT measured probabilities for the sampled cells.
# Rows: target Pyr/PV/SST/VIP; columns: source Pyr/PV/SST/VIP.
DEFAULT_TYPE_PROBABILITIES = np.array([
    [.20, .50, .45, .05],
    [.45, .35, .30, .05],
    [.35, .05, .05, .70],
    [.30, .05, .40, .05],
],dtype=float)


@dataclass
class Trajectory:
    rates_hz: np.ndarray
    rates_norm: np.ndarray
    logits: np.ndarray
    states: np.ndarray | None = None
    currents_pa: np.ndarray | None = None


def degree_preserving_rewire(mask, rng, swaps_per_edge=10):
    """Directed double-edge switches, preserving both in/out degrees exactly.

    No self-loops or multiedges. A finite switch chain is not claimed to sample
    uniformly from all degree-matched graphs; report accepted swaps/overlap and
    use multiple rewire seeds plus mixing sensitivity for a topology claim.
    """
    m = np.array(mask,dtype=bool,copy=True)
    edges = np.argwhere(m)
    count = len(edges)
    target = int(swaps_per_edge * count)
    accepted = attempted = 0
    if count >= 2:
        while accepted < target and attempted < max(100,50*target):
            attempted += 1
            p,q = rng.choice(count,2,replace=False)
            i,j = edges[p]
            k,l = edges[q]
            if i == k or j == l or i == l or k == j or m[i,l] or m[k,j]:
                continue
            m[i,j] = m[k,l] = False
            m[i,l] = m[k,j] = True
            edges[p] = [i,l]
            edges[q] = [k,j]
            accepted += 1
    if not (np.array_equal(m.sum(0),mask.sum(0)) and np.array_equal(m.sum(1),mask.sum(1))):
        raise AssertionError('Directed degree preservation failed')
    return m, {'accepted_swaps':accepted,'attempted_swaps':attempted,
               'requested_swaps':target,'edge_overlap_fraction':float(np.sum(m & mask)/max(1,count)),
               'uniform_null_sampling_claimed':False}


class Circuit:
    def __init__(self, params, cell_types, *, seed=42, in_dim=5,
                 identity='authentic',topology='type_prior',
                 recurrent_gain=.6,inhibitory_gain=1.5,input_gain=1.,
                 readout='pyramidal',input_mode='distributed',
                 type_probabilities=None,swaps_per_edge=10,
                 baseline_fraction=.05,input_scale_fraction=.25,
                 calibration=None,reference_rate_hz=100.,calibration_mode='reference'):
        self.reference_params = params
        self.cell_types = np.array(cell_types,dtype=str,copy=True)
        if self.cell_types.shape != (params.n,) or not np.all(np.isin(self.cell_types,CELL_TYPES)):
            raise ValueError('One known cell type is required for every neuron')
        self.params,self.identity_provenance = manipulate_identity(
            params,self.cell_types,identity,seed=int(seed))
        self.N,self.K = params.n,params.k
        self.a,self.beta,self.ds = self.params.a,self.params.beta,self.params.ds
        self.gamma_hz = self.params.gamma_hz
        if not np.isfinite(reference_rate_hz) or reference_rate_hz <= 0:
            raise ValueError('reference_rate_hz must be finite and positive')
        # One physical reference rate for every slot, independent of identity.
        # Per-cell ceiling normalization would partially cancel rate diversity.
        self.rate_scale_hz = np.full(self.N,float(reference_rate_hz))
        self.seed = int(seed)
        self.identity,self.topology = identity,topology
        self.in_dim = int(in_dim)
        if in_dim != 5:
            raise ValueError('This task uses five explicit input channels')
        for name, value in [('recurrent_gain', recurrent_gain),
                            ('inhibitory_gain', inhibitory_gain), ('input_gain', input_gain)]:
            if not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a finite nonnegative scalar')
        # 'reference' (v7): calibrate the intact bank and keep that offset/scale
        # for every identity control. A manipulated cell then sits at a different
        # operating point (shuffled networks fire ~2x faster), which confounds
        # identity with operating point. 'matched' (v11 default for controls):
        # rerun the same label-free current sweep on the manipulated bank, so each
        # cell's isolated baseline is again baseline_fraction of its own ceiling.
        if calibration_mode not in {'reference','matched'}:
            raise ValueError("calibration_mode must be 'reference' or 'matched'")
        self.calibration_mode = calibration_mode
        if calibration is None:
            calibration = calibrate_current(params,baseline_fraction,input_scale_fraction)
        self.reference_calibration_info = copy.deepcopy(calibration[2])
        if calibration_mode == 'matched' and identity != 'authentic':
            calibration = calibrate_current(self.params,baseline_fraction,input_scale_fraction)
        self.current_offset = np.array(calibration[0],copy=True)
        self.current_scale = np.array(calibration[1],copy=True)
        self.calibration_info = copy.deepcopy(calibration[2])
        if self.current_offset.shape != (self.N,) or self.current_scale.shape != (self.N,):
            raise ValueError('Calibration size mismatch')
        if not (np.all(np.isfinite(self.current_offset)) and np.all(np.isfinite(self.current_scale)) and np.all(self.current_scale>0)):
            raise ValueError('Invalid calibrated currents')
        self.signs = np.where(self.cell_types == 'Pyr',1.,-1.)
        probs = DEFAULT_TYPE_PROBABILITIES if type_probabilities is None else np.asarray(type_probabilities,dtype=float)
        if (probs.shape != (4,4) or not np.all(np.isfinite(probs))
                or np.any((probs<0)|(probs>1))):
            raise ValueError('type_probabilities must be a finite 4x4 matrix in [0,1]')
        self.type_probabilities = probs.copy()
        types = np.array([CELL_TYPES.index(ct) for ct in self.cell_types])
        edge_prob = probs[types[:,None],types[None,:]]
        graph_rng = rng_for(seed,'type_graph')
        reference_mask = graph_rng.random((self.N,self.N)) < edge_prob
        np.fill_diagonal(reference_mask,False)
        self.reference_mask = reference_mask.copy()
        weights_rng = rng_for(seed,'recurrent_magnitudes')
        magnitude = weights_rng.lognormal(-.5*.35**2,.35,(self.N,self.N))
        # Global mean incoming-degree scale is shared across all topology controls.
        norm = max(1.,float(reference_mask.sum()/self.N))
        ref_weights = magnitude * recurrent_gain/norm * reference_mask
        ref_weights[:,self.signs<0] *= inhibitory_gain
        self.rewire_info = {}
        if topology == 'type_prior':
            self.mask = reference_mask.copy()
            self.W_abs = ref_weights.copy()
        elif topology == 'degree_rewired':
            self.mask,self.rewire_info = degree_preserving_rewire(
                reference_mask,rng_for(seed,'rewiring'),swaps_per_edge)
            self.W_abs = np.zeros_like(ref_weights)
            wrng = rng_for(seed,'rewired_magnitudes')
            for j in range(self.N):
                # Preserve every source's outgoing magnitude multiset and sum.
                self.W_abs[self.mask[:,j],j] = wrng.permutation(ref_weights[reference_mask[:,j],j])
        elif topology == 'edge_count_random':
            possible = np.flatnonzero(~np.eye(self.N,dtype=bool))
            selected = rng_for(seed,'edge_count_graph').choice(possible,size=int(reference_mask.sum()),replace=False)
            self.mask = np.zeros((self.N,self.N),dtype=bool)
            self.mask.flat[selected] = True
            self.W_abs = magnitude * recurrent_gain/norm * self.mask
            self.W_abs[:,self.signs<0] *= inhibitory_gain
        else:
            raise ValueError(f'Unknown topology {topology}')
        self.edge_post,self.edge_pre = np.nonzero(self.mask)
        self.W_abs_initial = self.W_abs.copy()
        irng = rng_for(seed,'fixed_input')
        self.input_weights = irng.normal(0.,input_gain/np.sqrt(in_dim),(self.N,in_dim))
        if input_mode == 'vip_context':
            # Explicit afferent-routing hypothesis; never tied to topology mode.
            self.input_weights[:,2:4] = 0.
            vip = np.flatnonzero(self.cell_types == 'VIP')
            self.input_weights[vip,2:4] = irng.uniform(0.,input_gain,(len(vip),2))
        elif input_mode != 'distributed':
            raise ValueError(f'Unknown input mode {input_mode}')
        self.input_mode = input_mode
        if readout not in {'pyramidal','all'}:
            raise ValueError('readout must be pyramidal or all')
        self.readout_mask = (self.cell_types == 'Pyr') if readout == 'pyramidal' else np.ones(self.N,dtype=bool)
        orng = rng_for(seed,'readout')
        self.W_out = orng.normal(0.,.1/np.sqrt(max(1,self.readout_mask.sum())),self.N)*self.readout_mask
        self.b_out = 0.
        # Nonzero feedback can teach all postsynaptic types, matching BPTT's
        # plastic support. It remains frozen and is not a claim of weight transport.
        self.feedback = rng_for(seed,'feedback').normal(0.,1./np.sqrt(self.N),self.N)
        self.feedback_initial = self.feedback.copy()
        self.intrinsic_fingerprint = self.params.fingerprint()

    @property
    def W_rec(self):
        return self.W_abs * self.mask * self.signs[None,:]

    def activation_and_grad(self,z):
        return activation_and_grad(z,self.params)

    def initial_state(self,batch_size):
        return np.zeros((batch_size,self.N,self.K)),np.zeros((batch_size,self.N))

    def assert_constraints(self):
        if not np.all(np.isfinite(self.W_abs)) or not np.all(np.isfinite(self.W_out)) or not np.isfinite(self.b_out):
            raise FloatingPointError('Learned parameters are nonfinite')
        if np.any(self.W_abs < 0) or np.any(self.W_abs[~self.mask] != 0):
            raise AssertionError('Magnitude or structural mask constraint violated')
        if not np.array_equal(self.feedback,self.feedback_initial):
            raise AssertionError('Frozen feedback changed')
        if self.params.fingerprint() != self.intrinsic_fingerprint:
            raise AssertionError('Fitted intrinsic parameters changed')
        if np.any(self.W_out[~self.readout_mask] != 0):
            raise AssertionError('Readout support changed')

    def forward(self,X,*,current_noise=None,rate_multiplier=None,record_states=False,
                record_currents=False,initial_state=None,state_pulse=None):
        """Simulate exact GFR steps; current_noise is dimensionless drive noise.

        rate_multiplier is a [B,T,N] or broadcastable fraction in [0,1], applied
        after activation; it affects readout and all future feedback. It is an
        idealized rate clamp, not a model of opsin kinetics. state_pulse=(t,delta)
        adds a declared [B,N,K] perturbation before activation on one step.
        """
        X = np.asarray(X,dtype=float)
        if X.ndim != 3 or X.shape[-1] != self.in_dim or not np.all(np.isfinite(X)):
            raise ValueError('X must be finite [batch,time,in_dim]')
        b,tmax,_ = X.shape
        noise = np.zeros((b,tmax,self.N)) if current_noise is None else np.broadcast_to(np.asarray(current_noise,dtype=float),(b,tmax,self.N))
        multiplier = None if rate_multiplier is None else np.broadcast_to(np.asarray(rate_multiplier,dtype=float),(b,tmax,self.N))
        if not np.all(np.isfinite(noise)) or (multiplier is not None and (not np.all(np.isfinite(multiplier)) or np.any((multiplier<0)|(multiplier>1)))):
            raise ValueError('Nonfinite noise or rate multiplier outside [0,1]')
        h,r = self.initial_state(b) if initial_state is None else tuple(np.array(v,copy=True) for v in initial_state)
        rates = np.empty((b,tmax,self.N))
        logits = np.empty((b,tmax))
        states = np.empty((b,tmax,self.N,self.K)) if record_states else None
        currents = np.empty((b,tmax,self.N)) if record_currents else None
        W = self.W_rec
        for t in range(tmax):
            drive = X[:,t] @ self.input_weights.T + (r/self.rate_scale_hz) @ W.T + noise[:,t]
            current = self.current_offset + self.current_scale * drive
            h,r,_ = cell_step(self.params,h,r,current)
            if state_pulse is not None and t == state_pulse[0]:
                h = h + state_pulse[1]
                r,_ = self.activation_and_grad(h.mean(-1))
            if multiplier is not None:
                r = r * multiplier[:,t]
            rates[:,t] = r
            logits[:,t] = (r/self.rate_scale_hz) @ self.W_out + self.b_out
            if states is not None:
                states[:,t] = h
            if currents is not None:
                currents[:,t] = current
        return Trajectory(rates,rates/self.rate_scale_hz,logits,states,currents)

    def state_dict(self):
        return {'W_abs':self.W_abs.copy(),'W_out':self.W_out.copy(),'b_out':np.array(self.b_out),
                'intrinsic_fingerprint':np.array(self.intrinsic_fingerprint),
                'mask':self.mask.copy(),'input_weights':self.input_weights.copy(),
                'feedback':self.feedback.copy(),'current_offset':self.current_offset.copy(),
                'current_scale':self.current_scale.copy(),'rate_scale_hz':self.rate_scale_hz.copy(),
                'cell_types':self.cell_types.copy(),'signs':self.signs.copy(),
                'readout_mask':self.readout_mask.copy()}

    def load_state_dict(self,state):
        """Validate a complete checkpoint before changing any learned parameter.

        Construction arrays include biological slot signs and readout support;
        identical fitted cells and masks alone do not establish the same model.
        Missing construction fields are rejected, not silently inferred.
        """
        construction = ['mask','input_weights','feedback','current_offset','current_scale',
                        'rate_scale_hz','cell_types','signs','readout_mask']
        required = set(construction) | {'intrinsic_fingerprint','W_abs','W_out','b_out'}
        missing = required - set(state)
        if missing:
            raise ValueError(f'Checkpoint missing required fields: {sorted(missing)}')
        if str(state['intrinsic_fingerprint']) != self.intrinsic_fingerprint:
            raise ValueError('Checkpoint belongs to different fitted cells or manipulation')
        for name in construction:
            if not np.array_equal(state[name],getattr(self,name)):
                raise ValueError(f'Checkpoint construction mismatch: {name}')
        candidate = {}
        for name, shape in [('W_abs',(self.N,self.N)), ('W_out',(self.N,))]:
            if np.iscomplexobj(state[name]):
                raise ValueError(f'Checkpoint {name} must be real-valued')
            arr = np.array(state[name],dtype=np.float64,copy=True)
            if arr.shape != shape:
                raise ValueError(f'Checkpoint shape mismatch: {name}')
            if not np.all(np.isfinite(arr)):
                raise ValueError(f'Checkpoint {name} must be finite')
            candidate[name] = arr
        if np.iscomplexobj(state['b_out']):
            raise ValueError('Checkpoint b_out must be a real finite scalar')
        bias = np.asarray(state['b_out'],dtype=np.float64)
        if bias.shape != () or not np.isfinite(bias):
            raise ValueError('Checkpoint b_out must be a real finite scalar')
        if np.any(candidate['W_abs'] < 0) or np.any(candidate['W_abs'][~self.mask] != 0):
            raise ValueError('Checkpoint violates recurrent magnitude or mask constraints')
        if np.any(candidate['W_out'][~self.readout_mask] != 0):
            raise ValueError('Checkpoint violates readout support')
        if not np.array_equal(self.feedback,self.feedback_initial):
            raise AssertionError('Frozen feedback changed before checkpoint loading')
        if self.params.fingerprint() != self.intrinsic_fingerprint:
            raise AssertionError('Fitted intrinsic parameters changed before checkpoint loading')
        # All candidate and construction checks precede the first mutation.
        self.W_abs = candidate['W_abs']
        self.W_out = candidate['W_out']
        self.b_out = float(bias)
        self.assert_constraints()
