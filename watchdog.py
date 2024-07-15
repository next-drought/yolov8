import subprocess
import time
import logging

# Set up logging
logging.basicConfig(filename='watchdog.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

while True:
    logging.info("Starting home security script")
    process = subprocess.Popen(["python3", "homesecurity.py"])
    process.wait()
    logging.warning("Home security script stopped. Restarting in 5 seconds...")
    time.sleep(5)