"""
snowMaker

Core pipeline for extracting data from the Rocky Mountain snowpack
dataset hosting on hugging faces 
(https://huggingface.co/datasets/RMDig/rocky_mountain_snowpack).

Modules:
- pipeline: Class for piping snowpack data
- schema: Shared machinery for the metadata side tables
- pits, layers, cores, ect: One table each -- fields, domains and validation
- card: Dataset-card fragments generated from the table specs
"""

# Explicit imports from modules
import intake
import schema
import pits
import layers
import cores
import ect
import card
from pipeline import pipeline
from segmenter import colorSegmenter

# Define the public API
__all__ = [
    "intake",
    "schema",
    "pits",
    "layers",
    "cores",
    "ect",
    "card",
    "colorSegmenter",
    "pipeline"
]
