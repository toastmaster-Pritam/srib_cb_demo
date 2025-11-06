SRIB — Contextual Bandit Demo (Dockerized)

Lightweight demo / prototype that reproduces a Spotify-style impatient bandits pipeline and demonstrates a contextual-bandit experiment + a small UI.
This repository contains synthetic-data generation, the bandit training worker, an OBD pipeline worker, and a Streamlit UI — all orchestrated by docker-compose.


---

Quick start (Docker Compose)

Prerequisites:

Docker & Docker Compose (v1.27+ or Compose V2)

At least ~4GB free disk and enough RAM if you run all services

Recommended Python environment for local dev: 3.10+ (the container image uses the repo dockerfile)


Start everything (recommended):

# from repo root
docker compose up --build

This will:

1. Run data_gen service (container bandit_data_gen) which executes python data_generation.py to build the synthetic dataset and write to ./data.


2. Once the data is created, main_worker (bandit_main_worker) starts and runs python main.py (it waits for the synthetic dataset using wait_for_synthetic.sh).


3. obd_worker runs python obd_pipeline.py to process generated artifacts into ./obd.


4. streamlit runs streamlit run app.py exposing a small UI at http://localhost:8501.



If you prefer to run services individually (helpful for debugging), start them in sequence:

# generate data
docker compose run --rm data_gen

# start main worker (after data is ready)
docker compose run --rm main_worker

# start obd worker (in a different terminal)
docker compose run --rm obd_worker

# or start UI
docker compose run --service-ports streamlit
# then open: http://localhost:8501

Stop / tear down:

docker compose down


---

What each service does

docker-compose.yml defines four services:

data_gen (container bandit_data_gen)

Command: python data_generation.py

Purpose: create synthetic campaign / show traces (pickles) and save them under ./data (e.g. synthetic-data-train.pkl, synthetic-data-eval.pkl or your modified campaign filenames).

Mounts: repository root and ./data for output.

Runs once (restart: "no").


main_worker (container bandit_main_worker)

Command: ./wait_for_synthetic.sh && python main.py

Purpose: orchestrates experiments / runs bandit evaluation (your run_contextual_eval.py / main.py style code), saves trained agents & metrics under ./data (or configured artifact paths).

Depends on data_gen and uses wait_for_synthetic.sh to avoid race conditions.

Mounts ./obd to save processed outputs if relevant.


obd_worker (container bandit_obd_worker)

Command: python obd_pipeline.py

Purpose: post-process outputs (e.g., compute offline metrics, convert artifacts to OBD format, produce dashboards-ready files) and save to ./obd.

Runs once; used for offline analytics artifacts.


streamlit (container bandit_streamlit)

Command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0

Purpose: UI to explore results and interactively query a trained bandit (if UI code supports it). Exposes port 8501.

Healthcheck tries to curl the service.



---

Project layout & important files

(Your screenshot / repo includes these — quick mapping)

./admin_ui/                # (optional) admin UI assets
./data/                    # generated pickles & artifacts (mounted volume)
./impatient_bandits/       # Spotify-style library / belief/model code
./obd/random/all/          # sample OBD outputs
api.py                     # REST wrapper / small API (if present)
app.py                     # Streamlit web UI
docker-compose.yml         # orchestration (you pasted earlier)
dockerfile                 # docker image build instructions
main.py                    # main orchestration / worker entry-point
model_service.py           # model-serving helper / wrappers
obd_pipeline.py            # OBD post-processing pipeline
requirements.txt           # python deps
run_uvicorn.sh             # run server (if needed)
test.py                    # tests / sanity checks
wait_for_artifacts.sh      # helper for waiting on artifacts
wait_for_synthetic.sh      # helper for waiting on synthetic-data generation


---

Where outputs are saved

./data/ — synthetic datasets, trained agents, evaluation CSVs, pickles, plots. This folder is mounted into containers so you can see results on the host.

./obd/ — processed artifacts (optional) created by obd_pipeline.py.


Typical output filenames (examples used in code):

data/synthetic-data-train.pkl

data/synthetic-data-eval.pkl

data/trained_agent_<Belief>_n<...>.pkl

data/synthetic-data-eval.png (plots)

obd/* — OBD JSON/CSV artifacts.



---

How to inspect a saved trained model (quick)

After a run, trained bandit pickles are in ./data/ (filename pattern trained_agent_<name>_n<...>.pkl). Example usage (host, or inside a Python container):

import pickle, numpy as np

fname = "data/trained_agent_Progressive_n10.pkl"
with open(fname, "rb") as f:
    agent = pickle.load(f)

# Example context (intercept + features)
ctx = np.array([1.0, 0.45, 1, 0.2, 0])
action = agent.act(ctx, n_actions=1)[0]
print("Recommended campaign:", action)

Notes:

Ensure you load with the same codebase (same module/class paths) that was used to save the object, otherwise pickle may fail.

The bandit uses Thompson sampling — repeated act() calls for the same context can be stochastic by design.



---

How to get sample traces printed and pickled (developer)

data_generation.py or data_generation logic should:

1. Generate numpy arrays shaped (num_traces, dim) per campaign.


2. Save as a dict: { "campaign-000": arr, ... } via pickle.dump.


3. If you want to see sample rows while saving, add:



with open("data/synthetic-data-campaign-train.pkl", "wb") as f:
    pickle.dump(campaigns_dict, f)

# print sample rows (host console)
k = list(campaigns_dict.keys())[0]
print(k, campaigns_dict[k].shape)
print(campaigns_dict[k][:5, :10])  # first 5 traces, first 10 days


---

Environment variables & tuning

STREAMLIT_SERVER_HEADLESS, WAIT_TIMEOUT are set in the compose file. You can override via shell:


export WAIT_TIMEOUT=300
docker compose up --build

data_gen may take time if m is large (10k traces × 600 campaigns). Reduce m for faster runs while debugging.



---

Troubleshooting & logs

View container logs:


docker compose logs -f data_gen
docker compose logs -f bandit_main_worker
docker compose logs -f bandit_obd_worker
docker compose logs -f bandit_streamlit

If streamlit healthcheck fails, check port conflicts and set STREAMLIT_SERVER_HEADLESS=true properly in environment.

If pickle unpickle fails, ensure your local Python imports mirror the container module names (same package/module path).

If train step hangs, inspect wait_for_synthetic.sh timeout value and the data/ files.



---

Development tips

For fast iteration, run data_gen with smaller m (like 500) while debugging models.

Use docker compose run --rm <service> to run a single service interactively.

Mount a local Python venv and run main.py directly for step-by-step debugging without rebuilding the image.
