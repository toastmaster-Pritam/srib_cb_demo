import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable,Dict, List,Tuple

from .contextual_model import BayesianLinearModel

@dataclass
class LoggedStep:
    x:np.ndarray # context
    a:str #action (arm)
    r:float #reward scaler (1+ sum(traces))
    p_b:float #propensity pi_b(a|x)
    t:int #time index

class SoftmaxLoggingPolicy:
    """
    p_b(a|x)=softmax(beta* psi_a^T x) with minimum prob floor
    """

    def __init__(self,arms:List[str],dim:int,seed:int=123,beta:float=1.0,floor:float=0.01):
        self.rng=np.random.default_rng(seed)
        self.arms=list(arms)
        self.dim=dim
        self.beta=beta
        self.floor=floor

        self.psi = {a:self.rng.normal(scale=0.5,size=(dim,)) for a in arms} # random arm scoring vectors ( fixed)
    
    def probas(self,x:np.ndarray)->Dict[str,float]:
        logits = np.array([self.beta*(self.psi[a] @ x) for a in self.arms],dtype=float)
        logits -= logits.max()
        p=np.exp(logits)
        p /= p.sum()
        p=np.maximum(p,self.floor)
        p /= p.sum()
        return {a:float(p[i]) for i,a in enumerate(self.arms)}
    
    def act(self,x:np.ndarray)->Tuple[str,float]:
        pb=self.probas(x)
        pvec=np.array([pb[a] for a in self.arms],dtype=float)
        idx=int(self.rng.choice(len(self.arms),p=pvec))
        a=self.arms[idx]
        return a,float(pb[a])
    
def collect_logged_data(env,horizon:int,make_context:Callable[[],np.ndarray],generator_fn:Callable,reward_from_trace:Callable[[np.ndarray],float],logging_policy:SoftmaxLoggingPolicy)->List[LoggedStep]:
    data:List[LoggedStep]=[]

    for t in range(horizon):
        x=make_context()
        a,p_b=logging_policy.act(x)
        trace=env.step(a,t,n_samples=1,context=x,generator_fn=generator_fn)
        r=float(reward_from_trace(trace))
        data.append(LoggedStep(x=x,a=a,r=r,p_b=p_b,t=t))

    return data

class DirectMethodModel:
    """
    Per-arm linear reward model trained only on logged data for DM/Q-hat
    """
    def __init__(self,arms:List[str],dim:int,alpha:float=1.0,sigma2:float=0.25):
        self.models={a:BayesianLinearModel(dim,alpha,sigma2) for a in arms}
        self.arms=list(arms)
        self.dim=dim
    
    def fit(self,dataset:List[LoggedStep]):
        for step in dataset:
            self.models[step.a].update(step.x,step.r)
    
    def qhat(self,x:np.ndarray,a:str)->float:
        m=self.models[a].posterior_mean()
        return float(m @ x)
    
def greedy_eval_action(agent,x:np.ndarray)->str:
        return agent.greedy_action(x)

def _weight_clip(w:np.ndarray,clip:float)->np.ndarray:
    return np.minimum(w,clip) if clip is not None else w
    
def evaluate_offline(dataset:List[LoggedStep],arms:List[str],eval_action_fn:Callable[[np.ndarray],str],dm_model:DirectMethodModel,clip:float=10.0,snips:bool=True,rng:np.random.Generator=None,n_boot:int=500,)->Dict[str,Dict[str,float]]:
     if rng is None:
          rng=np.random.default_rng(7)
    
     n=len(dataset)
     x_arr= np.stack([s.x for s in dataset],axis=0)
     a_b=np.array([s.a for s in dataset])
     r=np.array([s.r for s in dataset],dtype=float)
     p_b=np.array([s.p_b for s in dataset],dtype=float)

     a_e=np.array([eval_action_fn(x) for x in x_arr])
     match=(a_b==a_e).astype(float)
     w=match/p_b
     w=_weight_clip(w,clip=clip)

     # DM Estimator for eval action and logged action
     q_e=np.array([dm_model.qhat(x,a) for x,a in zip(x_arr,a_e)],dtype=float)
     q_b=np.array([dm_model.qhat(x,a) for x,a in zip(x_arr,a_b)],dtype=float)

     #metrics
     ips=np.mean(w*r)
     snips_den=np.sum(w)
     snips_val=(np.sum(w*r)/snips_den) if snips_den>0 else np.nan
     dm=np.mean(q_e)
     dr=np.mean(q_e + w*(r-q_b))

    # Diagnostics
     ess = (snips_den ** 2) / np.sum(w ** 2) if snips_den > 0 else 0.0
     coverage = float(match.mean())

    # Bootstrap CIs (nonparametric)
     def _metric_from_idx(idx):
        wi = w[idx]; ri = r[idx]; qei = q_e[idx]; qbi = q_b[idx]
        ips_i = np.mean(wi * ri)
        snips_den_i = np.sum(wi)
        snips_i = (np.sum(wi * ri) / snips_den_i) if snips_den_i > 0 else np.nan
        dm_i = np.mean(qei)
        dr_i = np.mean(qei + wi * (ri - qbi))
        return ips_i, snips_i, dm_i, dr_i

     boot = np.empty((n_boot, 4), dtype=float)
     for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = _metric_from_idx(idx)

     def _ci(col):
        v = boot[:, col]
        v = v[~np.isnan(v)]
        return (float(np.mean(v)),
                float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)))

     ips_m, ips_lo, ips_hi = _ci(0)
     sn_m, sn_lo, sn_hi = _ci(1) if np.isfinite(snips_val) else (np.nan, np.nan, np.nan)
     dm_m, dm_lo, dm_hi = _ci(2)
     dr_m, dr_lo, dr_hi = _ci(3)

     return {
        "IPS":   {"value": float(ips),   "boot_mean": ips_m, "ci95": (ips_lo, ips_hi)},
        "SNIPS": {"value": float(snips_val), "boot_mean": sn_m, "ci95": (sn_lo, sn_hi)},
        "DM":    {"value": float(dm),    "boot_mean": dm_m,  "ci95": (dm_lo, dm_hi)},
        "DR":    {"value": float(dr),    "boot_mean": dr_m,  "ci95": (dr_lo, dr_hi)},
        "diagnostics": {"coverage": coverage, "ess": float(ess), "avg_weight": float(np.mean(w))}
     }




        



    



        
