import glob,re,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.stats import wilcoxon
def load(f):
    ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
    C={}
    for t in ea.Tags()['scalars']:
        m=re.match(r"F1/seed(\d+)_fold(\d+)$",t)
        if m: C[(int(m[1]),int(m[2]))]=np.array([e.value for e in ea.Scalars(t)])
    return C
R={}
for f in glob.glob("/mnt/user-data/uploads/sf2026/runs/tb/*/events*"):
    arm="emb" if "embedding" in f else "expr"; C=load(f)
    seeds=sorted({s for s,_ in C}); folds=sorted({k for _,k in C})
    lens=[len(v) for v in C.values()]
    out={"best":{},"loo":{},"loo_win":{},"loo_cf":{},"last10":{},"mean":{}}
    for s in seeds:
        for k in folds:
            c=C[(s,k)]; others=[C[(s,j)] for j in folds if j!=k]
            out["best"][(s,k)]=c.max()
            out["mean"][(s,k)]=c.mean()
            out["last10"][(s,k)]=c[-10:].mean()
            # LOO epoch: restricted to epochs where all other curves and this curve exist
            L=min(len(o) for o in others+[c])
            m=np.mean([o[:L] for o in others],0); e=int(m.argmax())
            out["loo"][(s,k)]=c[e]
            out["loo_win"][(s,k)]=c[max(0,e-5):e+6].mean()
            # carry-forward variant: pad curves with last value to max len
            M=max(len(o) for o in others+[c])
            pad=lambda x: np.concatenate([x,np.full(M-len(x),x[-1])])
            m2=np.mean([pad(o) for o in others],0); e2=int(m2.argmax())
            out["loo_cf"][(s,k)]=pad(c)[e2]
    R[arm]=out
    print(arm,"curves",len(C),"seeds",seeds,"epochs min/median/max",min(lens),int(np.median(lens)),max(lens))
for key in ["best","loo","loo_win","loo_cf","last10","mean"]:
    e=R["emb"][key]; x=R["expr"][key]
    ks=sorted(set(e)&set(x)); folds=sorted({k for _,k in ks})
    fe=np.array([np.mean([e[(s,k)] for s,kk in ks if kk==k]) for k in folds])
    fx=np.array([np.mean([x[(s,k)] for s,kk in ks if kk==k]) for k in folds])
    p=wilcoxon(fe,fx).pvalue if np.any(fe!=fx) else 1
    print(f"{key:8s} emb {fe.mean():.3f}  expr {fx.mean():.3f}  diff {fe.mean()-fx.mean():+.3f}  folds emb>expr {int((fe>fx).sum())}/8  wilcoxon p={p:.4f}")
