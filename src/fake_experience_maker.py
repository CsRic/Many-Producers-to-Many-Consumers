import torch
from dataclasses import dataclass
import time

@dataclass
class FakeExperience:
    data: torch.Tensor

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.data.to(device)

    def pin_memory(self):
        self.data.pin_memory()

class FakeExperienceMaker:
    def __init__(self,
                 produce_time = 3.0,
                 void_model = None,
                 data_shape = (4,4))->None:
        self.produce_time = produce_time
        self.void_model = void_model
        self.data_shape = data_shape
    def make_experience(self, void_input: torch.Tensor = None)-> FakeExperience:
        time.sleep(self.produce_time)
        data = torch.zeros(self.data_shape)#.to(torch.device(f'cuda:{torch.cuda.current_device()}'))
        return FakeExperience(data)