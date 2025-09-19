set -e

TIMEOUT = ${WAIT_TIMEOUT:-600}
SLEEP_INTERVAL=2
START = $(date +%s)

echo "[wait_for_synthetic] waiting upto $TIMEOUT seconds for synthetic-data files in /app/data ..."

while true; do
    if [ -f /app/data/synthetic-data-train.pkl] && [-f /app/data/synthetic-data-eval.pkl]; then
        echo "[wait_for_synthetic] found synthetic-data-train.pkl and synthetic-data-eval.pkl"
        exit 0
    fi

    NOW = $(date +%s)
    ELAPSED = $((NOW-START))
    if [ $ELAPSED -ge $TIMEOUT ]; then 
        echo "[wait_for_synthetic] Timeout ($TIMEOUT s) reached; synthetic data not present."
        echo "[wait_for_synthetic] Exiting with error code 1"
        exit 1
    fi 

    sleep $SLEEP_INTERVAL
done

