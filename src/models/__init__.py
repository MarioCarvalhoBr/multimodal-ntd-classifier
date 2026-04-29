import os
from dotenv import load_dotenv
load_dotenv()
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
print(f"CUDA_LAUNCH_BLOCKING: {os.getenv('CUDA_LAUNCH_BLOCKING')}")

print('Initializing models: Classifier and Trainer')
