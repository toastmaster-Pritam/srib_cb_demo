import pickle
import numpy as np

fname = "data/trained_agent_Progressive_n10.pkl"   # example
with open(fname, "rb") as f:
    agent = pickle.load(f)

test_context = np.array([1.0, 0.46, 0.33, 0.25, 0.1])
action = agent.act(test_context, n_actions=1)
print("Action:", action)