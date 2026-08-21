"""
Shared physical and model constants.
"""
import numpy as np

MATERIALS: dict[str, float] = {
    "Concrete": 2.3,
    "Steel":    7.86,
    "BPE":      1.04,
}

FLUX_SCALE: int = 200 * 200 * 50  # 2_000_000

E_ALL: np.ndarray = np.linspace(0.5, 249.5, 250)

MODEL_PKL_NAME: str = "model.pkl"
