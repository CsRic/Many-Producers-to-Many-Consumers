import torch
import time
from .fake_experience_buffer import FakeExperienceBuffer, FakeExperience

class FakeTrainer(torch.nn.Module):
    def __init__(self, fake_experience_buffer: FakeExperienceBuffer, name = "", train_time = 2.0, max_epoch = 10):
        self.buffer = fake_experience_buffer
        self.train_time = train_time
        self.max_epoch = max_epoch
        self.name = name

    def run_step(self):
        sample = self.buffer.sample()
        sample.to_device(torch.device(f"cuda:{torch.cuda.current_device()}"))
        # train...
        time.sleep(self.train_time)
    
    def fit(self):
        for i in range(self.max_epoch):
            self.run_step()
            print(f"{self.name}: trained {i + 1} / {self.max_epoch}")
    
    def get_buffer_length(self):
        return len(self.buffer)
    
    def put_experience(self, fake_experience:FakeExperience):
        self.buffer.append(fake_experience)