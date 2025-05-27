from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class Env(ABC):

    def __init__(self) -> None:
        self.obs_as_history: bool = False

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> str:
        pass
    
    @abstractmethod
    def step(self, action: Dict) -> Tuple:
        pass