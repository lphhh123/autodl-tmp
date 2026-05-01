#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt

METRICS = [
    "selected_total_scalar", "best_total_lastN", "use_cf_ratio", "trace_stage_probe_count",
    "top1_changed_stage_rate", "objective_std_lastN", "steps_per_1k_calls_lastN", "unique_sigs_per_1k_calls_lastN",
]

def parse_list(v):
    if isinstance(v, list): return v
    return [x for x in str(v).replace(',', ' ').split() if x]

def parse_budget_num(b):
    s=b.lower()
    if s.endswith('k'): return int(float(s[:-1])*1000)
    if s.endswith('m'): return int(float(s[:-1])*1000000)
    return int(s)

def latest_run_dir(root, exp, budget, inst, seed):
    d = root / f"{exp}-b{budget}_wT0p3_wC0p7_-{inst}" / f"seed{seed}"
    if not d.exists(): return None
    cands = [p for p in d.iterdir() if p.is_dir() and (p/'report.json').exists() and (p/'budget.json').exists()]
    if not cands: return None
    return sorted(cands, key=lambda p: p.stat().st_mtime)[-1]

def read_json(p):
    with open(p,'r',encoding='utf-8') as f: return json.load(f)

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/B')
    ap.add_argument('--out', default='outputs/B/_analysis/evidence_cd')
    ap.add_argument('--instance', default='cluster4')
    ap.add_argument('--budgets', nargs='+', default=['160k','240k'])
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--exp_prefixes', default='EXP-B2-std-budgetaware,EXP-B2-taos-style,EXP-B2-bc2cec,EXP-B2-bc2cec-nolong,EXP-B2-bc2cec-shiftm005,EXP-B2-bc2cec-shiftp005')
    args=ap.parse_args()

    root=Path(args.root); out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    budgets=parse_list(args.budgets)
    seeds=[int(x) for x in parse_list(args.seeds)]
    exps=parse_list(args.exp_prefixes)
    rows=[]
    for exp in exps:
        for b in budgets:
            for seed in seeds:
                rd=latest_run_dir(root,exp,b,args.instance,seed)
                row={'exp':exp,'budget':b,'seed':seed,'run_dir':str(rd) if rd else ''}
                for m in METRICS: row[m]=''
                row['actual_eval_calls']=''
                if rd:
                    rep=read_json(rd/'report.json')
                    bud=read_json(rd/'budget.json')
                    for m in METRICS: row[m]=rep.get(m, '')
                    row['actual_eval_calls']=bud.get('actual_eval_calls','')
                rows.append(row)

    csv_path=out/'evidence_cd.csv'
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def agg(exp, metric):
        xs=[]; ys=[]; es=[]
        for b in budgets:
            vals=[]
            for r in rows:
                if r['exp']==exp and r['budget']==b and r[metric] != '':
                    try: vals.append(float(r[metric]))
                    except: pass
            if vals:
                mean=sum(vals)/len(vals)
                std=(sum((v-mean)**2 for v in vals)/len(vals))**0.5
                xs.append(parse_budget_num(b)); ys.append(mean); es.append(std)
        return xs,ys,es

    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for exp in exps:
        x,y,e=agg(exp,'trace_stage_probe_count')
        if x: axes[0].errorbar(x,y,yerr=e,marker='o',label=exp)
        x,y,e=agg(exp,'use_cf_ratio')
        if x: axes[1].errorbar(x,y,yerr=e,marker='o',label=exp)
        x,y,e=agg(exp,'steps_per_1k_calls_lastN')
        if x: axes[2].errorbar(x,y,yerr=e,marker='o',label=f'{exp}:steps')
        x,y,e=agg(exp,'unique_sigs_per_1k_calls_lastN')
        if x: axes[2].errorbar(x,y,yerr=e,marker='x',linestyle='--',label=f'{exp}:uniq')
    axes[0].set_title('trace_stage_probe_count vs budget')
    axes[1].set_title('use_cf_ratio vs budget')
    axes[2].set_title('steps/unique_sigs per 1k calls vs budget')
    for ax in axes:
        ax.set_xlabel('budget (eval calls)'); ax.grid(True,alpha=0.3)
    axes[0].set_ylabel('count'); axes[1].set_ylabel('ratio'); axes[2].set_ylabel('metric')
    axes[2].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out/'evidence_cd.pdf')

    (out/'evidence_cd.md').write_text(
"""# C/D Process Evidence Notes\n\n- `trace_stage_probe_count` should increase with budget, indicating stage probes are firing along the budget axis.\n- `use_cf_ratio` indicates whether CF is actively used in the method implementation.\n- `steps_per_1k_calls_lastN` (and `unique_sigs_per_1k_calls_lastN`) indicate effective progress under fixed call budgets (calls_mode=eff rationale).\n""", encoding='utf-8')
    print(f'Wrote: {csv_path}, {out/"evidence_cd.pdf"}, {out/"evidence_cd.md"}')
