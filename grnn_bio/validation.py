"""Executable release validation. Reports actual failures and inconclusive checks."""
from pathlib import Path
import io,json,time,unittest
import numpy as np
from .data import load_dataset,sample_cells
from .circuit import Circuit
from .task import TaskConfig,generate_batch
from .learners import finite_difference_audit
from .io import write_json,environment,source_fingerprint,sha256_file


def validate_release(root,output):
    root=Path(root);start=time.perf_counter()
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    df=load_dataset(root/'model/gfr_dataset.json',root/'data/metadata.csv')
    p,rows=sample_cells(df,(12,4,2,2),seed=19)
    circuit=Circuit(p,rows.cell_type.to_numpy(),seed=47)
    batch=generate_batch(TaskConfig(t_fixation=2,t_stimulus=5,delay_bins=(3,5),t_decision=2),16,np.random.default_rng(918))
    audit=finite_difference_audit(circuit,batch,epsilon=1e-5,n_edges=32,seed=930)
    prov_path=root/'provenance/upstream/PROVENANCE_VERIFICATION.json'
    provenance=json.loads(prov_path.read_text()) if prov_path.exists() else None
    input_checks={}
    for name in ['model/gfr_dataset.json','data/metadata.csv']:
        actual=sha256_file(root/name)
        expected=provenance['files'][name]['sha256'] if provenance else None
        input_checks[name]={'sha256':actual,'matches_pinned_verified_input':actual==expected if expected else None}
    passed=result.wasSuccessful() and audit.get('status')=='pass' and all(x['matches_pinned_verified_input'] is True for x in input_checks.values())
    report={'status':'pass' if passed else 'fail_or_inconclusive','tests_run':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'test_log':stream.getvalue(),'finite_difference_actual_allen':audit,'input_integrity':input_checks,
            'elapsed_seconds':time.perf_counter()-start,'source_hash':source_fingerprint(root),'environment':environment(),
            'raw_patch_clamp_revalidation':'not performed; raw experimental sweeps are not bundled',
            'scope':'unit/integration/equation checks; this is not a biological performance or publication claim'}
    write_json(output,report)
    Path(output).with_suffix('.log').write_text(stream.getvalue(),encoding='utf-8')
    return {k:v for k,v in report.items() if k not in {'test_log','finite_difference_actual_allen'}} | {'finite_difference_status':audit.get('status')}
