"""Rebuild the r18 baseline from evaluation JSONs without overwriting raw evidence."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--checkpoints', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    a=ap.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    old=list(csv.DictReader((a.source/'matrix.csv').open(encoding='utf-8-sig')))
    groups=collections.defaultdict(list)
    for r in old:
        groups[(r['arch'],int(r['d']),int(r['seed']),r['precoder'],r['alpha_kappa'])].append(r)
    rows=[]; sources={}; diffs=collections.Counter()
    for group in groups.values():
        if len(group)>1:
            for k in group[0]:
                if len({r[k] for r in group})>1: diffs[k]+=1
    for arch in ('clnet','csinet','crnet','acrnet'):
        for d in (8,32):
            for seed in range(6):
                width=0.5 if arch=='acrnet' else 1.0
                ck=f'zoo-{arch}-d{d}-s{seed}'+('-w0.5' if arch=='acrnet' else '')
                cp=a.checkpoints/ck/'codec.pth'; sp=cp.with_name('summary.json')
                assert cp.exists(),cp
                summary=json.loads(sp.read_text(encoding='utf-8'))
                sources[str(cp)] = digest(cp)
                sources[str(sp)] = digest(sp)
                for pc,k in [('mrt',''),('rzf','1'),('rzf','16')]:
                    jp=a.source/(ck+'-'+pc+(f'-k{k}' if k else '')+'.json')
                    j=json.loads(jp.read_text(encoding='utf-8'))
                    assert j['split']=='test' and j['n_snapshots']==200 and j['interference_scope']=='full',jp
                    assert Path(j['data_dir']).name=='deepmimo_o1_r18',jp
                    assert j['precoder']['name']==pc and (not k or j['precoder']['alpha_kappa']==float(k)),jp
                    assert j['system_params']['power_per_ap_w']==0.1 and j['system_params']['subcarriers']==32,jp
                    se=j['se']['recon']; perf=j['se']['perfect']['sum_bps_hz']
                    r=dict(arch=arch,d=d,seed=seed,variant='zero-wu0',width=f'{width:g}',
                           bits=j['overhead']['total_bits_comb'],params=summary['parameters'],
                           macs_per_link=int(summary['macs_per_link_thop']),latency_ms='',
                           nmse_db=j['csi']['nmse_db'],precoder=pc,alpha_kappa=k,
                           se_recon=se['sum_bps_hz'],se_perfect=perf,
                           se_loss_pct=100*(perf-se['sum_bps_hz'])/perf,
                           p5_recon=se['p5_per_ue_bps_hz'],mean_ue=se['mean_per_ue_bps_hz'],
                           p5_per_sample=se['p5_per_sample_bps_hz'],
                           source_json=str(jp),checkpoint=str(cp))
                    assert all(math.isfinite(float(r[f])) for f in ('nmse_db','se_recon','se_perfect','p5_recon','p5_per_sample')),jp
                    assert r['bits']==(437 if d==8 else 1589),jp
                    for prior in groups[(arch,d,seed,pc,k)]:
                        for f,tol in [('se_recon',0.00006),('nmse_db',0.0006),('se_perfect',0.00006)]:
                            assert abs(float(prior[f])-r[f])<tol,(jp,f,prior[f],r[f])
                    # CPU latency was remeasured; retain all raw measurements in the audit, not one arbitrary value.
                    rows.append(r); sources[str(jp)]=digest(jp)
    assert len(rows)==144 and len(groups)==144
    with (a.outdir/'matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report={'raw_rows':len(old),'unique_configurations':len(rows),'extra_rows':len(old)-len(rows),
            'duplicate_differing_fields':dict(diffs),'latency_policy':'Excluded from canonical table: repeated CPU timing measurements; raw source retained.',
            'source_csv':str(a.source/'matrix.csv'),'source_csv_sha256':digest(a.source/'matrix.csv'),
            'sources_sha256':sources,'canonical_csv_sha256':digest(a.outdir/'matrix.csv')}
    (a.outdir/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='sources_sha256'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
