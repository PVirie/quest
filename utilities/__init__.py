from .package_install import install
from . import musique_classes, language_models, embedding_models
from .tokenizer import *
from datetime import datetime


def get_current_time_string():
    """
    Returns the current time in a formatted string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")