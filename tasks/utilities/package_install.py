import subprocess
import sys
import logging
import importlib.util
import regex as re
 
sys.path.append('/app/pip_modules')
logging.basicConfig(level=logging.INFO)

def install(package):
    # parse name and version (==, >=, etc.) from for example transformers>=4.45.1, transformers==4.45.1, transformers
    package_name = None
    package_condition = None
    package_version = None

    match = re.match(r"([a-zA-Z0-9_-]+)([<>=!]+)([0-9.]+)?", package)
    if match is not None:
        package_name = match.group(1)
        package_condition = match.group(2)
        package_version = match.group(3)
    else:
        package_name = package
    spec = importlib.util.find_spec(package_name)
    # matched is only when the package is already installed with the satisfying version
    matched = False
    if spec is not None:
        matched = True
        if package_condition is not None:
            matched = False
            # load module and check version
            module = importlib.import_module(package_name)
            spec.version = module.__version__
            if package_version is not None:
                if package_condition == "==":
                    matched = spec.version == package_version
                elif package_condition == ">=":
                    matched = spec.version >= package_version
                elif package_condition == "<=":
                    matched = spec.version <= package_version
                elif package_condition == ">":
                    matched = spec.version > package_version
                elif package_condition == "<":
                    matched = spec.version < package_version
                elif package_condition == "!=":
                    matched = spec.version != package_version

    if not matched:
        logging.warning(f"Installing: {package}")
        subprocess.check_call(["pip3", "install", package, "--target", "/app/pip_modules"])
