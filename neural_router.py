from typing import Dict, Type
from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class MistralModel(BaseModel):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B")
    
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0])

class CodeLlamaModel(BaseModel):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b")
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b")
    
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0])

class ModelRouter:
    MODELS: Dict[str, Type[BaseModel]] = {
        "mistral": MistralModel,
        "codellama": CodeLlamaModel,
        "custom": None  # Можно добавить свою модель
    }
    
    def __init__(self, model_type: str = "mistral"):
        self.model = self.MODELS[model_type]()
    
    def switch_model(self, new_type: str):
        if new_type in self.MODELS:
            self.model = self.MODELS[new_type]()
    
    def generate_response(self, prompt: str) -> str:
        return self.model.generate(prompt)
