"""
retrain_trigger.py  (Stage 1: decision layer only)

Purpose:
    Reads the latest drift check result (drift_status.json, written by
    drift_monitor.py), appends it to a running history (drift_history.json),
    and decides whether drift has been PERSISTENT enough (detected in the
    last N consecutive runs) to justify triggering a retrain.

    This version only makes the decision and prints it — it does NOT yet
    call the actual retraining code. That's added in Stage 2, once we've
    confirmed this decision logic behaves correctly on its own.

Why persistence, not a single run:
    A single drifted run could just be noise (one unusual batch). Requiring
    N consecutive drifted runs avoids retraining on a blip, mirroring how a
    real MLOps system would avoid overreacting to a single alert.
"""

import json
import os
from datetime import datetime

DRIFT_STATUS_FILE = "drift_status.json"
DRIFT_HISTORY_FILE = "drift_history.json"
PERSISTENCE_THRESHOLD = 2  # number of consecutive drifted runs required to trigger retraining


def load_latest_drift_status():
    if not os.path.exists(DRIFT_STATUS_FILE):
        raise FileNotFoundError(
            f"{DRIFT_STATUS_FILE} not found. Run drift_monitor.py first."
        )
    with open(DRIFT_STATUS_FILE, "r") as f:
        return json.load(f)


def load_history():
    if not os.path.exists(DRIFT_HISTORY_FILE):
        return []
    with open(DRIFT_HISTORY_FILE, "r") as f:
        return json.load(f)


def save_history(history):
    with open(DRIFT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def append_to_history(history, latest_status):
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_drift": latest_status["dataset_drift"],
        "drift_share": latest_status["drift_share"]
    }
    history.append(entry)
    return history


def is_retrain_needed(history, n=PERSISTENCE_THRESHOLD):
    if len(history) < n:
        return False
    recent = history[-n:]
    return all(entry["dataset_drift"] for entry in recent)


def main():
    latest_status = load_latest_drift_status()
    history = load_history()
    history = append_to_history(history, latest_status)
    save_history(history)

    print(f"Latest run: dataset_drift={latest_status['dataset_drift']}, "
          f"drift_share={latest_status['drift_share']}")
    print(f"Total runs recorded in history: {len(history)}")

    recent = history[-PERSISTENCE_THRESHOLD:]
    recent_flags = [e["dataset_drift"] for e in recent]
    print(f"Last {PERSISTENCE_THRESHOLD} run(s) drift flags: {recent_flags}")

    if is_retrain_needed(history):
        print(f"RETRAIN TRIGGERED — drift detected in the last "
              f"{PERSISTENCE_THRESHOLD} consecutive runs.")
        # Stage 2 will call the actual retraining logic here.
    else:
        print("Retrain NOT triggered — drift not yet persistent "
              "(or not currently detected). Continuing to monitor.")


if __name__ == "__main__":
    main()