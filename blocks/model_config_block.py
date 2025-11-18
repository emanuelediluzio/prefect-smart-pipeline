from typing import List

from prefect.blocks.core import Block
from pydantic import Field


class ModelConfigBlock(Block):
    """
    Block personalizzato per configurare il training del modello.
    Lo useremo per:
      - lista di n_estimators da testare
      - soglia RMSE per dire se il modello è "good" o "bad"
    """

    _block_type_name = "Model Config"

    n_estimators_list: List[int] = Field(
        default=[50, 100, 200],
        description="Lista di n_estimators da provare per la RandomForest.",
    )
    rmse_good_threshold: float = Field(
        default=0.3,
        description="Soglia di RMSE sotto cui il modello è considerato 'good'.",
    )
