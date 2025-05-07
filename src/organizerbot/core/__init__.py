"""
Core functionality package
"""
from .config import Config
from .processors import ImageProcessor, SelfTrainer
from .watcher import FileEventHandler, start_file_watcher 