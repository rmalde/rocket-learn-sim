import subprocess
import sys
from tqdm.rich import tqdm
import time

def launch_workers(num_workers):
    processes = []
    try:
        print("Initializing workers..")
        for _ in tqdm(range(num_workers)):
            p = subprocess.Popen(["python", "-m", "examples.default.worker"])
            processes.append(p)
            time.sleep(0.2)
        
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        for p in processes:
            try:
                p.terminate()
                p.wait()
            except Exception as e:
                print(f"Failed to terminate process {p.pid}: {e}")
    finally:
        sys.exit(0)

if __name__ == "__main__":
    launch_workers(20)
