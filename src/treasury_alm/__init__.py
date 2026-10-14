"""Treasury ALM analytics: deposit behaviour, NII forecasting and IRRBB stress testing."""

from pathlib import Path

__version__ = "0.1.0"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
CONFIG = ROOT / "config"
