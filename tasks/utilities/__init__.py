import subprocess
import sys
import logging
import importlib.util
 
sys.path.append('/app/pip_modules')
logging.basicConfig(level=logging.INFO)

def install(package_name):
    if importlib.util.find_spec(package_name) is None:
        logging.warning(f"Installing: {package_name}")
        subprocess.check_call(["pip3", "install", package_name, "--target", "/app/pip_modules"])
