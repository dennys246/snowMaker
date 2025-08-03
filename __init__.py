"""
snowMaker

Core pipeline for extracting data from the Rocky Mountain snowpack
dataset hosting on hugging faces 
(https://huggingface.co/datasets/dennys246/rocky_mountain_snowpack).

Modules:
- pipeline: Class for piping snowpack data
"""

# Explicit imports from modules
import intake
from pipeline import pipeline
from segmenter import colorSegmenter

# Define the public API
__all__ = [
    "intake",
    "colorSegmenter",
    "pipeline"
]