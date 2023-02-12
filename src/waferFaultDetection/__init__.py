import os
import sys
import logging
from datetime import datetime

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
dt = datetime.now()
dt_str = dt.strftime("%y%m%d_%H%M%S")
log_file = 'log_'+dt_str+'.txt'

log_filepath = os.path.join(log_dir,log_file)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger("waferFaultDetectionLogger")