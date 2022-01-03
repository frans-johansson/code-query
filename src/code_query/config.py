from types import SimpleNamespace
from typing import Any

from code_query.utils.serialize import yml_load


class RecursiveNamespace(SimpleNamespace):
    """
    Recursive extension of the `SimpleNamespace`
    adapted from https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8
    """
    @staticmethod
    def from_yml(file_name: str):
        # Read in configuration from .yml file
        config = yml_load(file_name)
        return RecursiveNamespace(**config)

    @staticmethod
    def map_entry(entry: Any) -> Any:
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


# Read config data from yml-files
DATA = RecursiveNamespace.from_yml("./config/data.yml")
TRAINING = RecursiveNamespace.from_yml("./config/training.yml")
WANDB = RecursiveNamespace.from_yml("./config/wandb.yml")
# It is more convenient for MODELS to be a plain dictionary for easy membership checking
MODELS = yml_load("./config/models.yml")
