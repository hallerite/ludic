from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any


class Env(ABC):

    def __init__(self) -> None:
        self.system_prompt: str = ""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> str:
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple:
        pass
    
    #@abstractmethod
    #def parse_action(self, action_str: str) -> Any:
    #    pass