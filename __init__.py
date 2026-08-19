"""
snowMaker

Core pipeline for extracting data from the Rocky Mountain snowpack
dataset hosting on hugging faces 
(https://huggingface.co/datasets/RMDig/rocky_mountain_snowpack).

Modules:
- pipeline: Class for piping snowpack data
- ect: Schema, domains and validation for the extended column test columns
"""

# Explicit imports from modules
import intake
import ect
from pipeline import pipeline
from segmenter import colorSegmenter

# Define the public API
__all__ = [
    "intake",
    "ect",
    "colorSegmenter",
    "pipeline"
]