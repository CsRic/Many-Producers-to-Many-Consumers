import torch
from dataclasses import dataclass
import time

@dataclass
class FakeExperience:
    data: torch.Tensor

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.data = self.data.to(device)
        return self

    def pin_memory(self):
        self.data.pin_memory()

class FakeExperienceMaker:
    def __init__(self,
                 produce_time = 3.0,
                 void_model = None,
                 data_shape = (4,4),
                 device = 'cpu')->None:
        self.produce_time = produce_time
        self.void_model = void_model
        self.data_shape = data_shape
        self.device = device
    def make_experience(self, void_input: torch.Tensor = None)-> FakeExperience:
        time.sleep(self.produce_time)
        data = torch.zeros(self.data_shape)
        return FakeExperience(data).to_device(self.device)