# SRIB — Contextual Bandit Demo (Dockerized)

Lightweight demo that reproduces a Spotify-style *impatient bandits* experiment pipeline.  
It demonstrates contextual-bandit training, evaluation, and visualization using synthetic data.  
All components are orchestrated with Docker Compose.

---

## 🧩 Project Overview

This project consists of four main services:

1. **data_gen** → Generates synthetic campaign data and stores it in `/data`.
2. **main_worker** → Trains contextual bandit models using the generated data.
3. **obd_worker** → Processes offline bandit diagnostics (OBD) and stores artifacts in `/obd`.
4. **streamlit** → Launches a Streamlit dashboard to visualize and interact with results.

Each service runs inside its own Docker container and shares data through mounted volumes.

---

## 🚀 Quick Start

### Prerequisites
- **Docker** and **Docker Compose v2+**
- Minimum **4GB RAM** and **2GB disk space**
- Internet access to pull required dependencies

### Steps to Run
```bash
# Clone the repository
git clone https://github.com/toastmaster-Pritam/srib_cb_demo.git
cd srib_cb_demo

# Build and start all services
docker compose up --build

This command will:

1. Run data_gen to create synthetic campaign datasets (.pkl files) under ./data/.


2. Start main_worker after data is ready to train the bandit models.


3. Run obd_worker to process offline bandit diagnostics.


4. Launch streamlit UI on http://localhost:8501.




---

🧠 What Each Service Does

🧩 data_gen

Runs python data_generation.py

Generates 600 synthetic campaigns, each with 10,000 traces of 59-day time-series data.

Saves:

data/synthetic-data-campaign-train.pkl

data/synthetic-data-campaign-eval.pkl



⚙️ main_worker

Runs ./wait_for_synthetic.sh && python main.py

Waits for synthetic data, then trains a contextual bandit model per belief.

Saves trained models in ./data/.


🧾 obd_worker

Runs python obd_pipeline.py

Processes model outputs, computes diagnostics, and writes to ./obd/.


📊 streamlit

Runs streamlit run app.py --server.port 8501

Hosts a Streamlit UI to visualize model performance and diagnostics.

Accessible at http://localhost:8501



---

🗂️ Repository Structure

.
├── admin_ui/                 # Optional admin interface
├── data/                     # Generated pickle data (mounted volume)
├── impatient_bandits/        # Bandit model definitions and utilities
├── obd/random/all/           # Processed OBD results
├── api.py                    # REST API endpoints
├── app.py                    # Streamlit dashboard entry point
├── docker-compose.yml        # Orchestration of all containers
├── dockerfile                # Image build instructions
├── main.py                   # Bandit training and evaluation logic
├── model_service.py          # Model serving utilities
├── obd_pipeline.py           # Offline diagnostics computation
├── requirements.txt          # Python dependencies
├── run_uvicorn.sh            # Optional API server runner
├── test.py                   # Unit or integration tests
├── wait_for_artifacts.sh     # Script to ensure artifacts exist
├── wait_for_synthetic.sh     # Script to ensure synthetic data exists
└── README.md


---

📁 Output Locations

Type	Path	Description

Synthetic data	./data/	Generated pickles for campaigns
Trained models	./data/	Pickled contextual bandit models
OBD artifacts	./obd/	Processed analytics and visualizations


Example saved files:

data/synthetic-data-campaign-train.pkl

data/synthetic-data-campaign-eval.pkl

data/trained_agent_<Belief>.pkl

obd/metrics_summary.json



---

🔍 Inspecting Saved Models

Once training completes, you can test a trained model manually:

import pickle, numpy as np

with open("data/trained_agent_Progressive.pkl", "rb") as f:
    agent = pickle.load(f)

context = np.array([1.0, 0.45, 1, 0.2, 0])  # Example context
action = agent.act(context, n_actions=1)[0]
print("Recommended campaign:", action)

> Note: Thompson sampling introduces randomness — the same context may yield slightly different actions.




---

⚙️ Environment Variables

Variable	Description	Default

WAIT_TIMEOUT	Max wait time (seconds) for synthetic data	1500
STREAMLIT_SERVER_HEADLESS	Run Streamlit in headless mode	true
STREAMLIT_SERVER_ENABLECORS	Disable CORS for Streamlit	false



---

🧰 Useful Commands

Run individual services

docker compose run --rm data_gen
docker compose run --rm main_worker
docker compose run --rm obd_worker
docker compose run --service-ports streamlit

Stop all containers

docker compose down

View logs

docker compose logs -f data_gen
docker compose logs -f main_worker
docker compose logs -f obd_worker
docker compose logs -f streamlit


---

⚠️ Troubleshooting

Issue	Possible Cause	Fix

streamlit fails to start	Port 8501 already in use	Change port in docker-compose.yml
No data in /data folder	data_gen failed	Check docker compose logs data_gen
Pickle load error	Different class path	Ensure same Python modules are used
Long runtime	Large m=10,000 traces	Reduce m in data_generation.py for faster testing



---

🧩 Development Notes

To modify synthetic data scale, edit parameters in data_generation.py:

dim = 59   # trace length
n = 600    # number of campaigns
m = 10000  # number of traces per campaign

To view sample generated traces before saving, insert:

print("Sample data:", data[0][:5, :10])

You can experiment locally without Docker using:

pip install -r requirements.txt
python data_generation.py
python main.py



---

🧾 License

MIT License © 2025
Maintained by toastmaster-Pritam
