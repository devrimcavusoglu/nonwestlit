from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CFG_DIR = PROJECT_ROOT / "cfg"
NEPTUNE_CFG = PROJECT_ROOT / "neptune.cfg"
