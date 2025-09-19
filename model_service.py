import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try importing experiment modules (tolerant)

from impatient_bandits import (
        ContextualBayesianBandit,
        ProgressiveBelief,
        EmpiricalDistribution,
        Environment,
        StickinessHelper,
        DirectMethodModel,
        evaluate_offline,
        LoggedStep
 )


# Logged entry
@dataclass
class SimpleLoggedStep:
    x: np.ndarray
    a: Any
    r: float
    p_b: float
    t: int
    ctx_id: Optional[int] = None

# Canonical feature pool (order matters)
CANONICAL_FEATURES = [
    "age",
    "device_mobile",
    "recent_eng",
    "premium",
    "time_of_day",
    "watch_time",
    "session_length",
    "is_weekend",
    "location_norm",
    "hour_of_day",
]

# Generators for canonical features (used to fill defaults)
def _gen_age(rng): return float(rng.uniform(0, 1))
def _gen_mobile(rng): return int(rng.binomial(1, 0.6))
def _gen_recent_eng(rng): return float(np.tanh(rng.exponential(1.0) / 2.0))
def _gen_premium(rng): return int(rng.binomial(1, 0.2))
def _gen_time_of_day(rng): return float(rng.uniform(0, 1))
def _gen_watch_time(rng): return float(np.clip(rng.exponential(0.2), 0.0, 1.0))
def _gen_session_length(rng): return float(np.clip(rng.exponential(0.5), 0.0, 1.0))
def _gen_is_weekend(rng): return int(rng.binomial(1, 0.28))
def _gen_location_norm(rng): return float(rng.integers(0, 100) / 99.0)
def _gen_hour_of_day(rng): return float(rng.integers(0, 23) / 23.0)

FEATURE_GENERATORS = {
    "age": _gen_age,
    "device_mobile": _gen_mobile,
    "recent_eng": _gen_recent_eng,
    "premium": _gen_premium,
    "time_of_day": _gen_time_of_day,
    "watch_time": _gen_watch_time,
    "session_length": _gen_session_length,
    "is_weekend": _gen_is_weekend,
    "location_norm": _gen_location_norm,
    "hour_of_day": _gen_hour_of_day,
}

def feature_label(name: str) -> str:
    return name.replace("_", " ").title()

# Safe vector functions
def _safe_align_vec(vec: np.ndarray, target_len: int) -> np.ndarray:
    v = np.asarray(vec).reshape(-1)
    if v.size == target_len:
        return v.astype(float)
    if v.size < target_len:
        out = np.zeros(target_len, dtype=float)
        out[: v.size] = v
        return out
    return v[:target_len].astype(float)

def _safe_posterior_mean_of_model(model, target_len: int) -> np.ndarray:
    try:
        pm = model.posterior_mean()
        arr = np.asarray(pm).reshape(-1)
    except Exception:
        return np.zeros(target_len, dtype=float)
    return _safe_align_vec(arr, target_len)

class ModelService:
    def __init__(
        self,
        train_pkl: str = "data/synthetic-data-train.pkl",
        eval_pkl: str = "data/synthetic-data-eval.pkl",
        context_dim: int = 4,
        pb_weight: float = 0.6,
        sigma2: float = 0.25,
        rng_seed: int = 1,
    ):
        self.train_pkl = train_pkl
        self.eval_pkl = eval_pkl
        self.context_dim = int(context_dim)
        self.pb_weight = float(pb_weight)
        self.sigma2 = float(sigma2)
        self.rng = np.random.default_rng(int(rng_seed))

        self.helper: Optional[StickinessHelper] = None
        self.env: Optional[Environment] = None
        self.item_traces: Optional[Dict[str, np.ndarray]] = None
        self.empirical_means: Optional[Dict[str, np.ndarray]] = None
        self.agent: Optional[ContextualBayesianBandit] = None

        # feature/instance state
        self.canonical_features = CANONICAL_FEATURES.copy()
        self.selected_features: List[str] = CANONICAL_FEATURES[:4]
        self.context_instances: List[Dict[str, Any]] = []

        # trace-related
        self.w: Optional[int] = None
        self.day_decay: Optional[np.ndarray] = None
        self.true_theta: Optional[Dict[str, np.ndarray]] = None

        # logging
        self.logged: List[SimpleLoggedStep] = []

        # init from pickles
        self._init_from_pickles()
        if len(self.context_instances) == 0:
            self.generate_random_instances(5)

    # -------------------
    # init/load
    # -------------------
    def _load_pickle(self, path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def _init_from_pickles(self):
        # load eval pkl to build env & empirical means
        if os.path.exists(self.eval_pkl):
            try:
                data_eval = self._load_pickle(self.eval_pkl)
                item_traces = {k: v.astype(float) for k, v in data_eval.items() if v is not None and len(v) > 0}
                if len(item_traces) > 0:
                    self.item_traces = item_traces
                    self.empirical_means = {k: np.mean(v, axis=0) for k, v in item_traces.items()}
                    # compute w and day_decay
                    any_arm = next(iter(self.empirical_means.values()))
                    self.w = any_arm.shape[0]
                    self.day_decay = np.linspace(1.0, 0.6, self.w)
                    # env
                    dists = {}
                    for uri, traces in item_traces.items():
                        seed_int = int(self.rng.integers(0, 2 ** 31 - 1))
                        dists[uri] = EmpiricalDistribution(traces.copy(), seed=seed_int)
                    self.env = Environment(dists)
                    # create true_theta mapping same style as main.py
                    self.true_theta = {uri: self.rng.normal(scale=0.8, size=(self.context_dim + 1,)) for uri in item_traces}
            except Exception as e:
                print("Warning: failed to init eval env from pkl:", e)

        # load training priors
        if os.path.exists(self.train_pkl):
            try:
                train_data = self._load_pickle(self.train_pkl)
                train_data = {k: v.astype(float) for k, v in train_data.items() if v is not None and len(v) > 0}
                if len(train_data) > 0:
                    self.helper = StickinessHelper.from_data(train_data)
            except Exception as e:
                print("Warning: failed to init helper from train pkl:", e)

    # -------------------
    # instance utilities
    # -------------------
    def _fill_instance_defaults(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        for f in self.canonical_features:
            if f not in instance:
                gen = FEATURE_GENERATORS.get(f)
                try:
                    instance[f] = gen(self.rng) if gen is not None else 0.0
                except Exception:
                    instance[f] = 0.0
        return instance

    def add_context_instance(self, ctx: Dict[str, Any]) -> int:
        inst = dict(ctx)
        inst = self._fill_instance_defaults(inst)
        self.context_instances.append(inst)
        return len(self.context_instances) - 1

    def update_context_instance(self, idx: int, ctx: Dict[str, Any]):
        if idx < 0 or idx >= len(self.context_instances):
            raise IndexError("context instance idx out of range")
        inst = dict(ctx)
        inst = self._fill_instance_defaults(inst)
        self.context_instances[idx] = inst

    def remove_context_instance(self, idx: int):
        if idx < 0 or idx >= len(self.context_instances):
            raise IndexError("context instance idx out of range")
        del self.context_instances[idx]

    def get_context_instances(self) -> List[Dict[str, Any]]:
        return [dict(c) for c in self.context_instances]

    def generate_random_instances(self, n: int = 5, selected_features: Optional[List[str]] = None) -> List[int]:
        if selected_features is None:
            selected_features = self.selected_features
        ids = []
        for _ in range(n):
            inst = {}
            for f in self.canonical_features:
                gen = FEATURE_GENERATORS.get(f)
                if f in selected_features and gen is not None:
                    inst[f] = gen(self.rng)
                else:
                    inst[f] = gen(self.rng) if gen is not None else 0.0
            idx = self.add_context_instance(inst)
            ids.append(idx)
        return ids

    # -------------------
    # feature selection
    # -------------------
    def get_feature_pool(self):
        return [{"name": f, "label": feature_label(f), "selected": f in self.selected_features} for f in self.canonical_features]

    def set_selected_features(self, selected: List[str]):
        selected = [s for s in selected if s in self.canonical_features]
        if len(selected) == 0:
            raise ValueError("At least one feature must be selected")
        self.selected_features = selected
        # retrofill existing instances so they have values for all canonical features
        for inst in self.context_instances:
            self._fill_instance_defaults(inst)

    def get_selected_features(self):
        return list(self.selected_features)

    # -------------------
    # agent lifecycle
    # -------------------
    def create_fresh_agent(self, belief_cls=ProgressiveBelief) -> ContextualBayesianBandit:
        if self.env is None:
            raise RuntimeError("No environment constructed (need eval pkl).")
        if self.helper is None:
            raise RuntimeError("No stickiness helper available (need train pkl).")
        beliefs = {}
        for uri in self.env.dists.keys():
            beliefs[uri] = belief_cls(
                prior_mvec=self.helper.prior_mvec.copy(),
                prior_cmat=self.helper.prior_cmat.copy(),
                noise_cmat=self.helper.noise_cmat.copy(),
                cov_estimator="fixed",
                seed=int(self.rng.integers(0, 2 ** 31 - 1)),
            )
        agent = ContextualBayesianBandit(beliefs, dim=self.context_dim + 1, alpha=1.0, sigma2=self.sigma2, pb_weight=self.pb_weight, seed=int(self.rng.integers(0, 2 ** 31 - 1)))
        self.agent = agent
        return agent

    def load_agent(self, path: str):
        with open(path, "rb") as fh:
            self.agent = pickle.load(fh)
        return self.agent

    def save_agent(self, path: str):
        if self.agent is None:
            raise RuntimeError("No agent to save")
        with open(path, "wb") as fh:
            pickle.dump(self.agent, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------
    # context vector building
    # -------------------
    def build_full_vector_from_instance(self, instance: Dict[str, Any]) -> np.ndarray:
        values = [1.0]
        for f in self.canonical_features:
            v = instance.get(f, 0.0)
            try:
                values.append(float(v))
            except Exception:
                try:
                    values.append(float(int(v)))
                except Exception:
                    values.append(0.0)
        vec = np.asarray(values, dtype=float).reshape(-1)
        vec = _safe_align_vec(vec, self.context_dim + 1)
        return vec
    
    # -------------------
    # context-dependent generator (MATCHES main.py)
    # -------------------
    def _generate_trace_for_arm(self, arm: str, context: np.ndarray, n_samples: int = 1):
        """
        Compute context-scaled day probabilities and sample binary traces.
        Mirrors main.py: base=empirical_means[arm], mu=true_theta[arm] @ context,
        scale = 0.5 * sigmoid(mu) + 0.5, probs = clip(base*scale*day_decay).
        """
        if self.empirical_means is None or self.day_decay is None or self.true_theta is None:
            # fallback to empirical sampling
            try:
                return self.env.dists[arm].sample(n=n_samples)
            except Exception:
                # fallback zeros
                w = self.w or 59
                return np.zeros((n_samples, w), dtype=float)
        base = self.empirical_means[arm]
        # ensure context shape
        x = np.asarray(context).reshape(-1)
        # align x to expected size (context_dim+1)
        x = _safe_align_vec(x, self.context_dim + 1)
        mu = float(self.true_theta[arm] @ x)
        scale = 0.5 * (1.0 / (1.0 + np.exp(-mu))) + 0.5
        probs = np.clip(base * scale * self.day_decay, 1e-6, 1 - 1e-6)
        samples = self.rng.binomial(1, probs, size=(n_samples, self.w)).astype(float)
        return samples

    def expected_reward_for_arm(self, arm: str, context: np.ndarray) -> float:
        if self.empirical_means is None or self.day_decay is None or self.true_theta is None:
            # fallback: empirical mean sum (context-agnostic)
            try:
                return 1.0 + self.empirical_means[arm].sum()
            except Exception:
                return 0.0
        base = self.empirical_means[arm]
        x = _safe_align_vec(np.asarray(context).reshape(-1), self.context_dim + 1)
        mu = float(self.true_theta[arm] @ x)
        scale = 0.5 * (1.0 / (1.0 + np.exp(-mu))) + 0.5
        probs = np.clip(base * scale * self.day_decay, 1e-6, 1 - 1e-6)
        return 1.0 + probs.sum()

    # -------------------
    # Thompson propensity estimator
    # -------------------
    def estimate_thompson_probas(self, context: np.ndarray, B: int = 500) -> Dict[str, float]:
        if self.agent is None:
            raise RuntimeError("No agent loaded")
        agent = self.agent
        arms = agent.arms
        K = len(arms)
        x = _safe_align_vec(np.asarray(context).reshape(-1), self.context_dim + 1)
        dim_x = x.size
        values = np.full((K, B), -np.inf, dtype=float)
        rng_local = np.random.default_rng(int(self.rng.integers(0, 2 ** 31 - 1)))
        for i, a in enumerate(arms):
            try:
                model = agent.models[a]
                Lambda = np.atleast_2d(np.asarray(model.Lambda))
                try:
                    cov = np.linalg.inv(Lambda)
                except Exception:
                    cov = np.linalg.pinv(Lambda)
                mean = _safe_posterior_mean_of_model(model, Lambda.shape[0])
            except Exception:
                mean = np.zeros(dim_x, dtype=float)
                cov = np.eye(dim_x, dtype=float) * 1e-6
            if cov.shape[0] != dim_x or cov.shape[1] != dim_x:
                if cov.shape[0] >= dim_x and cov.shape[1] >= dim_x:
                    cov = cov[:dim_x, :dim_x]
                else:
                    newcov = np.eye(dim_x, dtype=float) * 1e-6
                    newcov[: cov.shape[0], : cov.shape[1]] = cov
                    cov = newcov
                mean = _safe_align_vec(mean, dim_x)
            else:
                mean = _safe_align_vec(mean, dim_x)
            try:
                bel = agent.beliefs[a]
                if getattr(bel, "has_new_obs", False) or getattr(bel, "t_post", None) != getattr(agent, "t", None):
                    bel.compute_posterior(getattr(agent, "t", 0))
                pb_mean = float(getattr(bel, "mean", 0.0))
                pb_var = float(getattr(bel, "var", 0.0))
                pb_std = float(np.sqrt(max(pb_var, 1e-12)))
            except Exception:
                pb_mean, pb_std = 0.0, 1e-6
            try:
                thetas = rng_local.multivariate_normal(mean, cov, size=B)
            except Exception:
                thetas = np.tile(mean.reshape(1, -1), (B, 1))
            preds = thetas @ x
            pb_samples = pb_mean + pb_std * rng_local.standard_normal(size=B)
            pw = float(getattr(agent, "pb_weight", self.pb_weight))
            values[i, :] = preds + pw * pb_samples
        chosen = np.argmax(values, axis=0)
        counts = np.bincount(chosen, minlength=K)
        probas = counts.astype(float) / float(B)
        return {arms[i]: float(probas[i]) for i in range(K)}

    # -------------------
    # greedy action
    # -------------------
    def greedy_action(self, context_vec: np.ndarray, n_actions: int = 1) -> List[Any]:
        if self.agent is None:
            raise RuntimeError("No agent loaded")
        x = _safe_align_vec(np.asarray(context_vec).reshape(-1), self.context_dim + 1)
        arms = self.agent.arms
        values = []
        for a in arms:
            try:
                theta = _safe_posterior_mean_of_model(self.agent.models[a], x.size)
            except Exception:
                theta = np.zeros(x.size)
            theta = _safe_align_vec(theta, x.size)
            pred = float(theta @ x)
            pb_mean = 0.0
            try:
                bel = self.agent.beliefs[a]
                if getattr(bel, "has_new_obs", False) or getattr(bel, "t_post", None) != getattr(self.agent, "t", None):
                    bel.compute_posterior(getattr(self.agent, "t", 0))
                pb_mean = float(getattr(bel, "mean", 0.0))
            except Exception:
                pb_mean = 0.0
            val = pred + float(getattr(self.agent, "pb_weight", self.pb_weight)) * pb_mean
            values.append(val)
        idxs = np.argsort(values)[::-1][:n_actions]
        return [arms[i] for i in idxs]

    # -------------------
    # predict wrapper
    # -------------------
    def predict(self, instance: Optional[Dict[str, Any]] = None, vector: Optional[List[float]] = None, instance_id: Optional[int] = None, n_actions: int = 1, stochastic: bool = False) -> List[Any]:
        if instance_id is not None:
            instance = self.context_instances[instance_id]
        if vector is not None:
            vec = _safe_align_vec(np.asarray(vector).reshape(-1), self.context_dim + 1)
        elif instance is not None:
            vec = self.build_full_vector_from_instance(instance)
        else:
            if len(self.context_instances) > 0:
                vec = self.build_full_vector_from_instance(self.context_instances[self.rng.integers(0, len(self.context_instances))])
            else:
                tmp = {}
                for f in self.selected_features:
                    gen = FEATURE_GENERATORS.get(f)
                    tmp[f] = gen(self.rng) if gen is not None else 0.0
                vec = self.build_full_vector_from_instance(tmp)
        if stochastic:
            if self.agent is None:
                raise RuntimeError("No agent loaded")
            return self.agent.act(vec, n_actions=n_actions)
        else:
            return self.greedy_action(vec, n_actions=n_actions)

    # -------------------
    # batch_update
    # -------------------
    def batch_update(self, updates: List[Dict], log: bool = True) -> int:
        if self.agent is None:
            raise RuntimeError("No agent loaded")
        count = 0
        for u in updates:
            a = u["a"]
            trace = np.atleast_2d(np.asarray(u["trace"], dtype=float))
            ctx = u.get("context", None)
            instance_id = u.get("instance_id", None)
            if ctx is None and instance_id is not None:
                ctx = self.context_instances[instance_id]
            if ctx is None:
                ctx = {}
            vec = self.build_full_vector_from_instance(ctx)
            t = int(u.get("t", getattr(self.agent, "t", 0)))
            try:
                self.agent.update(a, trace, vec)
            except Exception:
                try:
                    self.agent.beliefs[a].update(trace, t)
                    self.agent.models[a].update(vec, float(1.0 + trace.sum()))
                except Exception:
                    pass
            p_b = float(u.get("p_b", 0.0))
            entry = SimpleLoggedStep(x=vec.copy(), a=a, r=float(1.0 + trace.sum()), p_b=p_b, t=t, ctx_id=instance_id)
            if log:
                self.logged.append(entry)
            count += 1
        return count

    # -------------------
    # simulate & log (uses context-dependent generator)
    # -------------------
    def simulate_and_log(self, horizon: int = 180, n_actions: int = 3, propensity_mc: int = 500, selected_instance_ids: Optional[List[int]] = None, context_mode: str = "random", fresh_agent: bool = True) -> Tuple[np.ndarray, np.ndarray, List[SimpleLoggedStep]]:
        if self.env is None:
            raise RuntimeError("No environment constructed - need eval pkl in place.")
        if self.helper is None:
            raise RuntimeError("No stickiness helper available - need train pkl.")
        if fresh_agent or self.agent is None:
            self.create_fresh_agent(ProgressiveBelief)
        agent = self.agent
        arms = list(self.env.dists.keys())
        logged_run: List[SimpleLoggedStep] = []
        regrets = []
        entropies = []
        rng_local = np.random.default_rng(int(self.rng.integers(0, 2 ** 31 - 1)))

        # build ctx sequence
        ctx_seq = []
        if selected_instance_ids:
            ids = [int(i) for i in selected_instance_ids]
            if context_mode == "round-robin":
                for i in range(horizon):
                    inst = self.context_instances[ids[i % len(ids)]]
                    ctx_seq.append(self.build_full_vector_from_instance(inst))
            else:
                for i in range(horizon):
                    inst = self.context_instances[ids[rng_local.integers(0, len(ids))]]
                    ctx_seq.append(self.build_full_vector_from_instance(inst))
        elif len(self.context_instances) > 0:
            if context_mode == "round-robin":
                for i in range(horizon):
                    inst = self.context_instances[i % len(self.context_instances)]
                    ctx_seq.append(self.build_full_vector_from_instance(inst))
            else:
                for i in range(horizon):
                    inst = self.context_instances[rng_local.integers(0, len(self.context_instances))]
                    ctx_seq.append(self.build_full_vector_from_instance(inst))
        else:
            for i in range(horizon):
                inst = {}
                for f in self.selected_features:
                    gen = FEATURE_GENERATORS.get(f)
                    inst[f] = gen(self.rng) if gen is not None else 0.0
                ctx_seq.append(self.build_full_vector_from_instance(inst))

        for t in range(horizon):
            x = np.asarray(ctx_seq[t]).reshape(-1)
            actions = agent.act(x, n_actions=n_actions)
            try:
                probas = self.estimate_thompson_probas(x, B=propensity_mc)
            except Exception:
                probas = {a: 1.0 / len(arms) for a in arms}
            for a in actions:
                p_b = max(probas.get(a, 0.0), 1e-12)
                # IMPORTANT: pass generator_fn so trace depends on context (like main.py)
                trace = self.env.step(a, t, n_samples=1, context=x, generator_fn=self._generate_trace_for_arm)
                trace = np.atleast_2d(trace)
                r = float(1.0 + trace.sum())
                entry = SimpleLoggedStep(x=x.copy(), a=a, r=r, p_b=p_b, t=t, ctx_id=None)
                logged_run.append(entry)
                try:
                    agent.update(a, trace, x)
                except Exception:
                    try:
                        agent.beliefs[a].update(trace, t)
                        agent.models[a].update(x, r)
                    except Exception:
                        pass
            agent.step()
            # Context-dependent oracle reward for regret
            try:
                true_rewards = np.array([self.expected_reward_for_arm(a, x) for a in actions])
                r_opt = max(self.expected_reward_for_arm(a, x) for a in arms)
            except Exception:
                true_rewards = np.zeros(len(actions), dtype=float)
                r_opt = 0.0
            regrets.append(float(np.mean(r_opt - true_rewards)))
            _, counts = np.unique(actions, return_counts=True)
            probs = counts / counts.sum()
            entropies.append(float(-np.sum(probs * np.log(probs + 1e-12))))
        self.logged.extend(logged_run)
        return np.array(regrets), np.array(entropies), logged_run

    def simulate_trials(self, n_trials: int = 3, horizon: int = 180, n_actions: int = 3, propensity_mc: int = 500, selected_instance_ids: Optional[List[int]] = None, context_mode: str = "random") -> Dict[str, Any]:
        all_regrets = []
        for i in range(n_trials):
            self.agent = None
            reg, ent, logs = self.simulate_and_log(horizon=horizon, n_actions=n_actions, propensity_mc=propensity_mc, selected_instance_ids=selected_instance_ids, context_mode=context_mode, fresh_agent=True)
            all_regrets.append(reg)
        arr = np.vstack(all_regrets)
        xs = np.arange(1, arr.shape[1] + 1)
        cumavg = np.cumsum(arr, axis=1) / xs
        mean = np.mean(cumavg, axis=0)
        std = np.std(cumavg, axis=0)
        return {"regrets": arr, "cumavg_mean": mean, "cumavg_std": std}

    def compute_ope(self, eval_agent=None, clip=10.0, snips=True, n_boot: int = 200):
        if len(self.logged) == 0:
            raise RuntimeError("No logged data available")
        dataset = self.logged
        arms = list(self.env.dists.keys()) if self.env is not None else list(self.agent.arms)
        dm_model = DirectMethodModel(arms=arms, dim=self.context_dim + 1, alpha=1.0, sigma2=self.sigma2)
        dm_model.fit(dataset)
        if eval_agent is None:
            eval_agent = self.agent
        def eval_action_fn(x):
            x = np.asarray(x).reshape(-1)
            return self.greedy_action(x, n_actions=1)[0]
        metrics = evaluate_offline(dataset=dataset, arms=arms, eval_action_fn=eval_action_fn, dm_model=dm_model, clip=clip, snips=snips, n_boot=n_boot)
        return metrics