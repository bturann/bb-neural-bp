import hashlib,json
from pathlib import Path
import numpy as np
r=Path('results/project_assessment/20261110-review')
cycle=Path('results/m6_cycle100_study/20261109T114431070320Z/summary.json')
paths=[cycle,Path('results/m6_aux_report/20261109T135141505262Z/report.md'),Path('results/m6_scaling_selector/20261109-approved/confirmation/summary.json'),Path('results/m6_cycle100_planning/20261110T015309813879Z/analysis.json'),Path('../01_Research_Plan_Learned_BP_Decoder.pdf')]
s=json.loads(cycle.read_text());points=[]
for p in s['points']:
 b=p['failures'][p['ordinary_name']];a=p['failures']['selected'];n=p['shots']
 points.append(dict(p=p['p'],shots=n,ordinary_fallback=b,learned_fallback=a,absolute_reduction_pp=100*(b-a)/n,relative_reduction_percent=100*(b-a)/b,
   illustrative_common_fallback_quantile_ordinary=1-.01/(b/n),illustrative_common_fallback_quantile_learned=1-.01/(a/n)))
tails=[]
for name,root in [('longer',Path('results/m6_long_bp_latency/20261109T124523000569Z')),('strong',Path('results/m6_strong_cascade_latency/20261109T130204474636Z'))]:
 for i in [0,1]:
  path=root/f'raw-p{i}.jsonl';paths.append(path)
  raw=[json.loads(x) for x in path.read_text().splitlines()]
  for method in ['learned100','bp100','native100']+(['ordinary'] if name=='strong' else []):
   rows=[x for x in raw if x['method']==method and not x['warmup']]
   if not rows:continue
   times=np.array([x['total_us'] for x in rows]);top=[x for x in rows if x['total_us']>=np.percentile(times,99)]
   tails.append(dict(block=name,p=[.001,.003][i],method=method,measured_cases=len(rows),fallback_requests=sum(x['fallback_called'] for x in rows),
     empirical_tail_cases=len(top),empirical_tail_fallback_cases=sum(x['fallback_called'] for x in top)))
evidence=dict(stage1=points,tail_checks=tails,interpretation='Derived read-only from saved evidence. Quantile mapping is an illustrative common-fallback/separated-components approximation, not a prediction or extra measurement.',
 scope='Project viability assessment only; no new training, decoding, benchmark or final-test data',new_experimental_cases=0,
 input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(r/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
(r/'derive_evidence.py').write_bytes(Path(__file__).read_bytes())
print('Derived evidence saved; no new experimental cases.')
