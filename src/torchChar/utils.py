from pathlib import Path

def get_models_dir():
    mpath = Path(__file__).parent / "../../models"
    return mpath.resolve().absolute()

def get_data_dir():
    mpath = Path(__file__).parent / "../../data"
    return mpath.resolve().absolute()