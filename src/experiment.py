#!/usr/bin/env python3
"""
AD-IRT-Mem: Psychometric Token Allocation for OCR-Memory Systems
================================================================
FINAL production experiment — all audit blockers resolved.

Fixes from audit round 2:
  [B1] Type 3 fonts → pdf.fonttype=42, ps.fonttype=42
  [B2] Table 4 values → auto-generated LaTeX from CSV
  [B3] CV/test overlap → disjoint splits: train(0-39), cv(40-59), eval(60-79)
  [B4] Runtime → timed and reported

Author: Jung Min Kang (gangjeongmin23@gmail.com)
ORCID: 0009-0007-9599-2792
"""

import argparse, os, json, time
import numpy as np
from scipy.stats import spearmanr, sem
from scipy.special import expit
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. DATA GENERATION
# ============================================================

class MemoryBank:
    def __init__(self, n_ep=40, n_cpe=12, difficulty_gap=3.0, seed=42):
        rng = np.random.RandomState(seed)
        self.n_ep, self.n_cpe = n_ep, n_cpe
        self.N = n_ep * n_cpe
        self.difficulty_gap = difficulty_gap
        self.feat_names = ["action_complexity","state_transitions",
                           "tool_depth","error_density","temporal_recency"]
        self.true_w = np.array([1.5, 0.8, 1.2, 2.0, 0.5])
        self.n_feat = 5
        self.X = rng.uniform(0, 1, (self.N, self.n_feat))
        raw = self.X @ self.true_w + rng.normal(0, 0.3, self.N)
        s = max(raw.std(), 1e-6)
        self.b = (raw - raw.mean()) / s * (difficulty_gap / 2)
        self.a = np.clip(0.5 + rng.exponential(0.6, self.N), 0.3, 3.5)
        self.episode = np.repeat(np.arange(n_ep), n_cpe)
        self.n_families = 5
        self.family = self.episode % self.n_families
        self.info = np.abs(self.b) * self.a


class QuerySet:
    def __init__(self, n_q, mb, seed=42):
        rng = np.random.RandomState(seed)
        self.n_q = n_q
        self.theta = rng.normal(0, 1.2, n_q)
        self.rel_p = expit(mb.a[None,:] * (self.theta[:,None] - mb.b[None,:]))
        self.rel_bin = (rng.uniform(0,1,(n_q, mb.N)) < self.rel_p).astype(float)
        self.info_val = self.rel_bin * mb.info[None,:]


# ============================================================
# 2. OCR ACCURACY MODEL
# ============================================================

def ocr_accuracy(r, b_j, a_j):
    return expit(0.7 * a_j * (3.0 * r - 1.0 - 0.6 * b_j))


# ============================================================
# 3. IRT FITTING
# ============================================================

def fit_2pl(R, max_iter=200, lr=0.003):
    mask = ~np.isnan(R)
    nq, nc = R.shape
    theta, b, la = np.zeros(nq), np.zeros(nc), np.zeros(nc)
    for _ in range(max_iter):
        a = np.exp(np.clip(la, -2, 2))
        for i in range(nq):
            oj = np.where(mask[i])[0]
            if len(oj)==0: continue
            p = expit(a[oj]*(theta[i]-b[oj]))
            theta[i] += lr*(np.sum(a[oj]*(R[i,oj]-p)) - 0.01*theta[i])
        for j in range(nc):
            oi = np.where(mask[:,j])[0]
            if len(oi)==0: continue
            aj = np.exp(np.clip(la[j],-2,2))
            p = expit(aj*(theta[oi]-b[j]))
            res = R[oi,j]-p
            b[j]  += lr*(np.sum(-aj*res) - 0.005*b[j])
            la[j] += lr*(np.sum((theta[oi]-b[j])*res*aj) - 0.01*la[j])
    return theta, b, np.exp(np.clip(la, -2, 2))


# ============================================================
# 4. ALLOCATION STRATEGIES (with B<64n fix)
# ============================================================

def _distribute(cands, budget, weights):
    n = len(cands)
    if budget < 64 or n == 0:
        return {}
    if budget < n * 64:
        n_aff = int(budget // 64)
        top = np.argsort(-weights)[:n_aff]
        surplus = budget - n_aff * 64
        sw = weights[top]; sw = sw/(sw.sum()+1e-12)
        return {cands[i]: np.clip(surplus*sw[idx]/336,0,1) for idx,i in enumerate(top)}
    surplus = budget - n*64
    return {c: np.clip(surplus*weights[i]/336,0,1) for i,c in enumerate(cands)}

def alloc_uniform(cands, budget, **_):
    n=len(cands); return _distribute(cands,budget,np.ones(n)/max(n,1)) if n else {}

def alloc_recency(cands, budget, mb, **_):
    if not len(cands): return {}
    r=mb.X[cands,4]+0.1; return _distribute(cands,budget,r/r.sum())

def alloc_topk(cands, budget, rel_scores, **_):
    if not len(cands): return {}
    order=np.argsort(-rel_scores[cands])
    alloc,rem={},budget
    for idx in order:
        c=cands[idx]
        if rem>=400: alloc[c]=1.0; rem-=400
        elif rem>=64: alloc[c]=np.clip((rem-64)/336,0,1); rem=0; break
    return alloc

def alloc_fisher(cands, budget, mb, theta_i, est_b, est_a, **_):
    n=len(cands)
    if n==0: return {}
    fi=np.array([est_a[c]**2*expit(est_a[c]*(theta_i-est_b[c]))
                 *(1-expit(est_a[c]*(theta_i-est_b[c]))) for c in cands])
    w=(fi+1e-8)/(fi.sum()+1e-8*n)
    return _distribute(cands,budget,w)

def alloc_adirt(cands, budget, mb, theta_i, est_b, est_a, w_param=0.0, **_):
    n=len(cands)
    if n==0: return {}
    fi=np.array([est_a[c]**2*expit(est_a[c]*(theta_i-est_b[c]))
                 *(1-expit(est_a[c]*(theta_i-est_b[c]))) for c in cands])
    n_fam=len(np.unique(mb.family[cands]))
    alpha=expit(w_param*(n_fam/mb.n_families - 0.4))
    fw=(fi+1e-8)/(fi.sum()+1e-8*n)
    uw=np.ones(n)/n
    bl=(1-alpha)*fw + alpha*uw; bl/=bl.sum()
    return _distribute(cands,budget,bl)


# ============================================================
# 5. EVALUATION
# ============================================================

def compute_irr(alloc, qi, mb, qs, rng):
    total=qs.info_val[qi].sum()
    if total<1e-8: return 0.0
    rec=0.0
    for c,r in alloc.items():
        if rng.uniform()<ocr_accuracy(r, mb.b[c], mb.a[c]):
            rec+=qs.info_val[qi,c]
    return rec/total


# ============================================================
# 6. INNER CV FOR w  [B3 FIX: uses disjoint CV queries 40-59]
# ============================================================

def select_w(mb, qs, et, eb, ea, n_cand=25, budget=2500, seed=0):
    """3-fold inner CV on queries 40-59 (disjoint from eval 60-79)."""
    w_grid=np.arange(-3.0,3.5,0.5)
    cv_start, cv_end = 40, 60
    fold_size = (cv_end-cv_start)//3
    best_w, best_s = 0.0, -np.inf
    for w in w_grid:
        fs=[]
        for fold in range(3):
            vs=cv_start+fold*fold_size; ve=vs+fold_size
            irrs=[]
            for qi in range(vs, min(ve, cv_end)):
                rng=np.random.RandomState(seed*100000+qi*100+int((w+3)*10))
                noisy=qs.rel_p[qi]+rng.normal(0,0.12,mb.N)
                cands=np.argsort(-noisy)[:n_cand]
                al=alloc_adirt(cands,budget,mb=mb,theta_i=et[qi%len(et)],
                               est_b=eb,est_a=ea,w_param=w)
                irrs.append(compute_irr(al,qi,mb,qs,rng))
            fs.append(np.mean(irrs))
        ms=np.mean(fs)
        if ms>best_s: best_s=ms; best_w=w
    return best_w


# ============================================================
# 7. EXPERIMENTS
# ============================================================

def experiment_1(out, n_seeds=5):
    """Main comparison. [B3 FIX]: eval queries 60-79, disjoint from CV."""
    budgets=[0.10,0.20,0.30,0.50,0.70]
    dgaps=[1.0,2.0,3.0,4.0,5.0]
    METHODS=["uniform","recency","topk","fisher","adirt"]
    n_cand, n_eval_start, n_eval_end = 25, 60, 80  # [B3 FIX]
    results=[]

    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds}", end="", flush=True)
        for dgap in dgaps:
            base=seed*100+int(dgap*10)
            mb=MemoryBank(difficulty_gap=dgap, seed=base)
            qs=QuerySet(80, mb, seed=base+50)

            # IRT training on queries 0-39
            rng_h=np.random.RandomState(seed*1000+int(dgap*100))
            hist=np.full((40, mb.N), np.nan)
            for i in range(40):
                obs=rng_h.choice(mb.N,80,replace=False)
                for j in obs:
                    hist[i,j]=1.0 if rng_h.uniform()<qs.rel_p[i,j] else 0.0
            et,eb,ea=fit_2pl(hist)
            obs_items=np.where(~np.isnan(hist).all(axis=0))[0]
            b_rho=spearmanr(mb.b[obs_items],eb[obs_items])[0] if len(obs_items)>10 else 0

            # [B3 FIX]: CV on queries 40-59
            sel_w=select_w(mb,qs,et,eb,ea,seed=seed)

            for bf in budgets:
                budget=int(n_cand*400*bf)
                m_irr={m:[] for m in METHODS}
                # [B3 FIX]: Eval on queries 60-79 (disjoint)
                for qi in range(n_eval_start, n_eval_end):
                    rng_q=np.random.RandomState(seed*100000+qi+int(dgap*1000))
                    noisy=qs.rel_p[qi]+rng_q.normal(0,0.12,mb.N)
                    cands=np.argsort(-noisy)[:n_cand]
                    ti=et[qi%40]
                    allocs={
                        "uniform": alloc_uniform(cands,budget),
                        "recency": alloc_recency(cands,budget,mb=mb),
                        "topk":    alloc_topk(cands,budget,rel_scores=noisy),
                        "fisher":  alloc_fisher(cands,budget,mb=mb,theta_i=ti,est_b=eb,est_a=ea),
                        "adirt":   alloc_adirt(cands,budget,mb=mb,theta_i=ti,est_b=eb,est_a=ea,w_param=sel_w),
                    }
                    for mi,m in enumerate(METHODS):
                        rng_e=np.random.RandomState(seed*999999+qi*10+mi)
                        m_irr[m].append(compute_irr(allocs[m],qi,mb,qs,rng_e))

                for m in METHODS:
                    results.append(dict(seed=seed,difficulty_gap=dgap,budget_frac=bf,
                                        method=m,mean_irr=np.mean(m_irr[m]),
                                        std_irr=np.std(m_irr[m]),b_rho=b_rho,selected_w=sel_w))
            print(".",end="",flush=True)
        print()

    df=pd.DataFrame(results)
    df.to_csv(os.path.join(out,"exp1_main.csv"),index=False)
    return df


def experiment_2(out, n_seeds=5):
    """Sparsity effect."""
    sp_levels=[0.0,0.2,0.4,0.6,0.8,0.9]; budget=3000; n_cand=25
    results=[]
    for seed in range(n_seeds):
        mb=MemoryBank(difficulty_gap=3.0,seed=seed*200)
        qs=QuerySet(80,mb,seed=seed*200+50)
        for sp in sp_levels:
            rng_h=np.random.RandomState(seed*2000+int(sp*100))
            hist=np.full((40,mb.N),np.nan)
            for i in range(40):
                n_obs=max(5,int(mb.N*(1-sp)*0.2))
                obs=rng_h.choice(mb.N,min(n_obs,mb.N),replace=False)
                for j in obs: hist[i,j]=1.0 if rng_h.uniform()<qs.rel_p[i,j] else 0.0
            et,eb,ea=fit_2pl(hist)
            obs_items=np.where(~np.isnan(hist).all(axis=0))[0]
            b_rho=spearmanr(mb.b[obs_items],eb[obs_items])[0] if len(obs_items)>10 else 0
            sel_w=select_w(mb,qs,et,eb,ea,seed=seed)
            for m in ["uniform","fisher","adirt"]:
                irrs=[]
                for qi in range(60,80):
                    rng_q=np.random.RandomState(seed*300000+qi+int(sp*1000))
                    noisy=qs.rel_p[qi]+rng_q.normal(0,0.12,mb.N)
                    cands=np.argsort(-noisy)[:n_cand]; ti=et[qi%40]
                    if m=="uniform": al=alloc_uniform(cands,budget)
                    elif m=="fisher": al=alloc_fisher(cands,budget,mb=mb,theta_i=ti,est_b=eb,est_a=ea)
                    else: al=alloc_adirt(cands,budget,mb=mb,theta_i=ti,est_b=eb,est_a=ea,w_param=sel_w)
                    rng_e=np.random.RandomState(seed*888888+qi)
                    irrs.append(compute_irr(al,qi,mb,qs,rng_e))
                results.append(dict(seed=seed,sparsity=sp,method=m,mean_irr=np.mean(irrs),b_rho=b_rho))
    df=pd.DataFrame(results)
    df.to_csv(os.path.join(out,"exp2_sparsity.csv"),index=False)
    return df


def experiment_3(out, n_seeds=5):
    """LLTM recovery."""
    results=[]
    for seed in range(n_seeds):
        mb=MemoryBank(difficulty_gap=3.0,seed=seed*600)
        qs=QuerySet(60,mb,seed=seed*600+50)
        rng=np.random.RandomState(seed*600+888)
        hist=np.full((50,mb.N),np.nan)
        for i in range(50):
            obs=rng.choice(mb.N,120,replace=False)
            for j in obs: hist[i,j]=1.0 if rng.uniform()<qs.rel_p[i%60,j] else 0.0
        _,eb,_=fit_2pl(hist)
        obs=np.where(~np.isnan(hist).all(axis=0))[0]
        X,y=mb.X[obs],eb[obs]
        w_hat=np.linalg.solve(X.T@X+0.5*np.eye(5),X.T@y)
        yp=X@w_hat
        r2=1-np.sum((y-yp)**2)/max(np.sum((y-y.mean())**2),1e-8)
        results.append(dict(seed=seed,w_rank_rho=spearmanr(mb.true_w,w_hat)[0],
                            b_pred_rho=spearmanr(y,yp)[0],r2=r2,est_w=w_hat.tolist()))
    df=pd.DataFrame(results)
    df.to_csv(os.path.join(out,"exp3_lltm.csv"),index=False)
    return df


# ============================================================
# 8. FIGURES  [B1 FIX: Type 42 fonts]
# ============================================================

def make_figures(df1, df2, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # [B1 FIX]: Force Type 42 (TrueType) fonts, no Type 3
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams.update({"font.size":10,"axes.labelsize":11,"axes.titlesize":12,"figure.dpi":200})

    C={"uniform":"#888888","recency":"#999999","topk":"#4DBEEE","fisher":"#D95319","adirt":"#0072BD"}
    L={"uniform":"Uniform","recency":"Recency","topk":"Top-k Full","fisher":"IRT Fisher","adirt":"AD-IRT-Mem"}
    M={"uniform":"s","recency":"^","topk":"v","fisher":"o","adirt":"D"}

    # Fig 1: IRR by budget
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
    for idx,dg in enumerate([1.0,3.0,5.0]):
        ax=axes[idx]
        sub=df1[df1["difficulty_gap"]==dg]
        agg=sub.groupby(["method","budget_frac"]).agg(m=("mean_irr","mean"),s=("mean_irr",sem)).reset_index()
        for m in ["uniform","recency","topk","fisher","adirt"]:
            d=agg[agg["method"]==m].sort_values("budget_frac")
            ax.errorbar(d["budget_frac"]*100,d["m"],yerr=d["s"],color=C[m],marker=M[m],
                       label=L[m],capsize=3,lw=1.5,ms=5)
        ax.set_xlabel("Token Budget (% of max)"); ax.set_title(f"Difficulty Gap D = {dg:.0f}")
        ax.grid(True,alpha=0.3)
        if idx==0: ax.set_ylabel("Information Recovery Ratio")
        if idx==2: ax.legend(fontsize=7,loc="lower right")
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(out,"fig1_irr_budget.pdf"),bbox_inches="tight")
    plt.savefig(os.path.join(out,"fig1_irr_budget.png"),bbox_inches="tight")
    plt.close()

    # Fig 2: Gap interaction (corrected title)
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    for m,col,lab in [("fisher","#D95319","Fisher $-$ Uniform"),("adirt","#0072BD","AD-IRT-Mem $-$ Uniform")]:
        ds,gs=[],[]
        for dg in [1.0,2.0,3.0,4.0,5.0]:
            sub=df1[(df1["difficulty_gap"]==dg)&(df1["budget_frac"]==0.3)]
            u=sub[sub["method"]=="uniform"]["mean_irr"].mean()
            v=sub[sub["method"]==m]["mean_irr"].mean()
            ds.append(v-u); gs.append(dg)
        ax.plot(gs,ds,"o-",color=col,label=lab,lw=1.5)
    ax.axhline(0,color="gray",ls="--",alpha=0.5)
    ax.set_xlabel("Difficulty Gap (range of $b_j$)")
    ax.set_ylabel("$\\Delta$IRR vs. Uniform")
    ax.set_title("IRT Advantage Peaks at Low-to-Moderate\nDifficulty Gaps (Budget = 30%)")
    ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(out,"fig2_gap_interaction.pdf"),bbox_inches="tight")
    plt.savefig(os.path.join(out,"fig2_gap_interaction.png"),bbox_inches="tight")
    plt.close()

    # Fig 3: Sparsity
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4.2))
    sagg=df2.groupby(["method","sparsity"]).agg(m=("mean_irr","mean"),s=("mean_irr",sem)).reset_index()
    for m in ["uniform","fisher","adirt"]:
        d=sagg[sagg["method"]==m].sort_values("sparsity")
        ax1.errorbar(d["sparsity"]*100,d["m"],yerr=d["s"],color=C[m],marker="o",label=L[m],capsize=3,lw=1.5)
    ax1.set_xlabel("Historical Sparsity (%)"); ax1.set_ylabel("Mean IRR")
    ax1.set_title("IRR vs. Historical Data Sparsity"); ax1.legend(fontsize=8); ax1.grid(True,alpha=0.3)
    brho=df2.groupby("sparsity").agg(m=("b_rho","mean"),s=("b_rho",sem)).reset_index()
    ax2.errorbar(brho["sparsity"]*100,brho["m"],yerr=brho["s"],color="#0072BD",marker="s",capsize=3,lw=1.5)
    ax2.set_xlabel("Historical Sparsity (%)"); ax2.set_ylabel("$b$ Recovery $\\rho$")
    ax2.set_title("IRT Difficulty Recovery vs. Sparsity"); ax2.set_ylim(0,1); ax2.grid(True,alpha=0.3)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(out,"fig3_sparsity.pdf"),bbox_inches="tight")
    plt.savefig(os.path.join(out,"fig3_sparsity.png"),bbox_inches="tight")
    plt.close()
    print("  Figures saved (Type 42 fonts).")


# ============================================================
# 9. AUTO-GENERATE LATEX TABLE SNIPPETS [B2 FIX]
# ============================================================

def generate_latex_tables(df1, out):
    """Write LaTeX table data directly from CSV to ensure consistency."""
    # Table 3: IRR by budget
    agg=df1.groupby(["method","budget_frac"]).agg(m=("mean_irr","mean")).reset_index()
    lines=[]
    for m in ["uniform","recency","topk","fisher","adirt"]:
        label={"uniform":"Uniform","recency":"Recency","topk":"Top-$k$ Full",
               "fisher":"IRT Fisher","adirt":"AD-IRT-Mem"}[m]
        vals=[]
        for bf in [0.10,0.20,0.30,0.50,0.70]:
            d=agg[(agg["method"]==m)&(agg["budget_frac"]==bf)]
            vals.append(f".{d['m'].values[0]*1000:.0f}" if len(d) else "---")
        bold = "\\textbf{" + label + "}" if m=="adirt" else label
        lines.append(f"{bold} & {' & '.join(vals)} \\\\")
    with open(os.path.join(out,"table3_data.tex"),"w") as f:
        f.write("\n".join(lines))

    # Table 4: IRR by difficulty gap at 30%
    sub=df1[df1["budget_frac"]==0.3]
    gagg=sub.groupby(["method","difficulty_gap"]).agg(m=("mean_irr","mean")).reset_index()
    lines=[]
    for m in ["uniform","topk","fisher","adirt"]:
        label={"uniform":"Uniform","topk":"Top-$k$ Full",
               "fisher":"IRT Fisher","adirt":"AD-IRT-Mem"}[m]
        vals=[]
        for dg in [1.0,2.0,3.0,4.0,5.0]:
            d=gagg[(gagg["method"]==m)&(gagg["difficulty_gap"]==dg)]
            vals.append(f".{d['m'].values[0]*1000:.0f}" if len(d) else "---")
        lines.append(f"{label} & {' & '.join(vals)} \\\\")
    with open(os.path.join(out,"table4_data.tex"),"w") as f:
        f.write("\n".join(lines))

    print("  LaTeX table snippets saved.")


# ============================================================
# 10. MAIN
# ============================================================

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out_dir",default="./output")
    parser.add_argument("--seeds",type=int,default=5)
    args=parser.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    N=args.seeds

    print("="*72)
    print("  AD-IRT-Mem: Psychometric Token Allocation for OCR-Memory")
    print(f"  Seeds={N}, Output={args.out_dir}")
    print("  Disjoint splits: train(0-39), CV(40-59), eval(60-79)")
    print("="*72)

    t0=time.time()

    print("\n--- Experiment 1: Main Comparison ---")
    df1=experiment_1(args.out_dir,n_seeds=N)

    # Print tables from CSV
    agg=df1.groupby(["method","budget_frac"]).agg(m=("mean_irr","mean"),s=("mean_irr",sem)).reset_index()
    print("\n  Table 3 (from CSV):")
    for m in ["uniform","recency","topk","fisher","adirt"]:
        row=f"    {m:12s}"
        for bf in [0.10,0.20,0.30,0.50,0.70]:
            d=agg[(agg["method"]==m)&(agg["budget_frac"]==bf)]
            row+=f"  {d['m'].values[0]:.3f}" if len(d) else "  N/A"
        print(row)

    # Table 4
    sub=df1[df1["budget_frac"]==0.3]
    gagg=sub.groupby(["method","difficulty_gap"]).agg(m=("mean_irr","mean")).reset_index()
    print("\n  Table 4 (from CSV, budget=30%):")
    for m in ["uniform","topk","fisher","adirt"]:
        row=f"    {m:12s}"
        for dg in [1.0,2.0,3.0,4.0,5.0]:
            d=gagg[(gagg["method"]==m)&(gagg["difficulty_gap"]==dg)]
            row+=f"  {d['m'].values[0]:.4f}" if len(d) else "  N/A"
        print(row)

    # Win analysis
    print("\n  Win Analysis:")
    for bl in ["uniform","recency","topk","fisher"]:
        wins,total,deltas=0,0,[]
        for _,grp in df1.groupby(["seed","budget_frac","difficulty_gap"]):
            a=grp[grp["method"]=="adirt"]["mean_irr"].values
            b=grp[grp["method"]==bl]["mean_irr"].values
            if len(a) and len(b):
                total+=1; d=a[0]-b[0]; deltas.append(d)
                if d>1e-6: wins+=1
        print(f"    vs {bl:12s}: {wins}/{total} ({100*wins/max(total,1):.1f}%) "
              f"Δ={np.mean(deltas):+.4f}")

    w_dist=df1[df1["method"]=="adirt"].groupby(["seed","difficulty_gap"])["selected_w"].first()
    print(f"\n  Selected w values: {sorted(w_dist.unique())}")

    print("\n--- Experiment 2: Sparsity ---")
    df2=experiment_2(args.out_dir,n_seeds=N)

    print("\n--- Experiment 3: LLTM ---")
    df3=experiment_3(args.out_dir,n_seeds=N)
    print(f"  Weight rank ρ: {df3['w_rank_rho'].mean():.3f}±{df3['w_rank_rho'].sem():.3f}")
    print(f"  b pred ρ:      {df3['b_pred_rho'].mean():.3f}±{df3['b_pred_rho'].sem():.3f}")
    print(f"  R²:            {df3['r2'].mean():.3f}±{df3['r2'].sem():.3f}")

    print("\n--- Generating Figures ---")
    make_figures(df1,df2,args.out_dir)

    print("\n--- Generating LaTeX Tables ---")
    generate_latex_tables(df1, args.out_dir)

    elapsed=time.time()-t0
    b_rho_mean = float(df1["b_rho"].mean())

    summary={"irr_by_method":df1.groupby("method")["mean_irr"].mean().to_dict(),
             "b_recovery_rho":b_rho_mean,
             "lltm_r2":float(df3["r2"].mean()),
             "lltm_weight_rank_rho":float(df3["w_rank_rho"].mean()),
             "lltm_b_pred_rho":float(df3["b_pred_rho"].mean()),
             "wall_clock_seconds":round(elapsed,1),
             "n_seeds":N,
             "query_splits":"train(0-39), CV(40-59), eval(60-79)"}
    with open(os.path.join(args.out_dir,"summary.json"),"w") as f:
        json.dump(summary,f,indent=2)

    print(f"\n✓ Complete in {elapsed:.0f}s. Outputs in {args.out_dir}/")

if __name__=="__main__":
    main()
