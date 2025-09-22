
import torch
import torch.nn.functional as F
import os

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained, Llama
from tokenizer import Tokenizer


class LlamaPretrainingModel(torch.nn.Module):
	def __init__(self, config):
		super(LlamaPretrainingModel, self).__init__()

		# Automatically detect whether to train from scratch or load checkpoint
		if os.path.exists(config.pretrained_model_path):
			print(f"Loading checkpoint from {config.pretrained_model_path}")
			self.llama = load_pretrained(config.pretrained_model_path)
		else:
			# print("Training from scratch - creating new model")
			# Create default config for training from scratch
			llama_config = LlamaConfig(
				vocab_size=32000,
				dim=512,
				dropout=0.0,
				n_layers=8,
				n_heads=8,
				n_kv_heads=8,
				max_seq_len=2048,
				layer_norm_eps=1e-5
			)
			self.llama = Llama(llama_config)
			self.llama.init_weights()  # Initialize weights randomly

		# For pretraining, we update all parameters
		for param in self.llama.parameters():
			param.requires_grad = True

	def forward(self, input_ids):
		# For pretraining, we just return the raw logits from the LLaMA model
		# This method is kept for compatibility but direct access to self.llama is used in training
		logits, _ = self.llama(input_ids, targets=input_ids)
		return F.log_softmax(logits, dim=-1)