import os
from dotenv import load_dotenv

from utils.logger import logger


load_dotenv()
logger.info(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
logger.info(f"CUDA_LAUNCH_BLOCKING: {os.getenv('CUDA_LAUNCH_BLOCKING')}")

logger.info('Initializing models: Classifier and Trainer')
