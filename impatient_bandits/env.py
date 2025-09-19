# impatient_contextual/env.py
import collections

class Environment:
    def __init__(self, dists):
        """
        dists: dict mapping item ID -> Distribution object (EmpiricalDistribution etc.)
        """
        self.dists = dists
        self.history = collections.defaultdict(list)

    def step(self, action, t, n_samples=1, context=None, generator_fn=None):
        """Return an n_samples x w trace matrix for the chosen action.
        If generator_fn provided, call it as generator_fn(action, context, n_samples).
        Else call self.dists[action].sample(n=n_samples).
        """
        self.history[t].append(action)
        if generator_fn is not None:
            return generator_fn(action, context, n_samples)
        return self.dists[action].sample(n=n_samples)

    def reset(self):
        for dist in self.dists.values():
            try:
                dist.reset()
            except Exception:
                pass
        self.history = collections.defaultdict(list)