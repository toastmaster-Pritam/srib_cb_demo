set -e

TIMEOUT = ${WAIT_TIMEOUT:-1500}
SLEEP_INTERVAL=3
START = $(date +%s)

echo "[wait_for_artifacts] waiting upto $TIMEOUT seconds for artifacts to be available..."

while true; do
    HAVE_SYNTH = 0
    HAVE_ARTIFACTS = 0
    if [ -f /app/data/synthetic-data-train.pkl ] && [ -f /app/data/synthetic-data-eval.pkl ]; then
        HAVE_SYNTH = 1
    fi

    if [ -f /app/data/experiment_results.pkl  ]; then
        HAVE_ARTIFACTS = 1
    fi

    if ls /app/data/trained_agent_*.pkl >/dev/null 2>&1; then
        HAVE_ARTIFACTS = 1
    fi

    if [ $HAVE_SYNTH -eq 1 ] && [ $HAVE_ARTIFACTS -eq 1 ]; then
        echo "[wait_for_artifacts] All synthetic data and artifacts are available. Proceeding..."
        exit 0
    fi

    NOW = $(date +%s)
    ELAPSED = $((NOW-START))
    if [ $ELAPSED -ge $TIMEOUT ]; then 
        echo "[wait_for_artifacts] Timeout ($TIMEOUT s) reached; proceeding anyway."
        exit 0
    fi 

    sleep $SLEEP_INTERVAL
done

