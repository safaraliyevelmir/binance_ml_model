import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.main import N_TRIALS, STUDY_NAME, optuna_main

nWorkers: int = 4

cpuCount = os.cpu_count() or 1
rfPerWorker = max(1, cpuCount // nWorkers)
storage = "sqlite:///optuna.db"


def splitTrials(total: int, workers: int) -> list[int]:
    """Distribute `total` trials across `workers` as evenly as possible."""
    base, rem = divmod(total, workers)
    return [base + (1 if i < rem else 0) for i in range(workers)]


def worker(workerId: int, trialsForMe: int) -> dict[str, int]:
    """Single worker process: runs its share of trials against the shared study."""
    optuna_main(
        n_trials=trialsForMe,
        study_name=STUDY_NAME,
        optuna_n_jobs=1,        # this worker = 1 thread; parallelism is across processes
        rf_n_jobs=rfPerWorker,
        storage=storage,
    )
    return {"worker": workerId, "trials": trialsForMe}


if __name__ == "__main__":
    shares = splitTrials(N_TRIALS, nWorkers)
    print(
        f"[launch] cpuCount={cpuCount}  workers={nWorkers}  "
        f"rfPerWorker={rfPerWorker}  totalTrials={N_TRIALS}  "
        f"split={shares}  storage={storage}"
    )

    with ProcessPoolExecutor(max_workers=nWorkers) as pool:
        futures = [
            pool.submit(worker, wid, shares[wid])
            for wid in range(nWorkers)
            if shares[wid] > 0
        ]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                print(f"[launch] worker {result['worker']} done ({result['trials']} trials)")
            except Exception as e:
                print(f"[launch] worker FAILED: {e!r}")

    print("[launch] all workers finished")
