
from prometheus_client import start_http_server, Gauge
import time
train_time = Gauge('training_run_seconds', 'Time taken to train model')
def simulate():
    start_http_server(8001)
    while True:
        train_time.set(time.time() % 100)
        time.sleep(5)
if __name__ == "__main__":
    simulate()
