from abc import ABC, abstractmethod
from typing import Any
from .fake_experience_maker import FakeExperience
import torch
import torch.multiprocessing as multiprocessing
class FakeExperienceBufferBase(ABC):
    def __init__(self, limit: int = 0) -> None:
        super().__init__()
        self.limit = limit

    @abstractmethod
    def append(self, experience: FakeExperience) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def sample(self) -> FakeExperience:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class FakeExperienceBuffer(FakeExperienceBufferBase):
    def __init__(self, total_limit: int = 0, gpu_limit: int = 0, cpu_offload: bool = True) -> None:
        self.cpu_offload = cpu_offload
        self.total_limit = total_limit
        # self.gpu_limit = gpu_limit
        # self.target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.items = multiprocessing.Queue(total_limit)

    @torch.no_grad()
    def append(self, experience: FakeExperience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device('cpu'))
        self.items.put(experience)
        
    def clear(self) -> None:
        self.items.join()
        self.items.close()
        self.items = multiprocessing.Queue(self.total_limit)

    @torch.no_grad()
    def sample(self) -> FakeExperience:
        # block if not available
        return self.items.get()
    
    def __len__(self) -> int:
        return self.items.qsize()