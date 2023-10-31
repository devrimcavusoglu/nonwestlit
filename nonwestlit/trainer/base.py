from abc import ABC
from typing import Optional

from transformers import TrainingArguments


class BaseTrainingPipeline(ABC):
	def __init__(self, model_name_or_path: str, data_path: str, bnb_quantization: Optional[str] = None, **kwargs):
		self.model_name_or_path = model_name_or_path
		self.data_path = data_path
		self.bnb_quantization = bnb_quantization

		self.init_model_and_tokenizer(model_name_or_path, bnb_quantization)
		self.training_args = TrainingArguments(**kwargs)
		self.__post_init__()

	def __post_init__(self):
		pass

	def __call__(self, *args, **kwargs):
		pass

	def init_model_and_tokenizer(self, model_name_or_path, bnb_quantization):
		pass


