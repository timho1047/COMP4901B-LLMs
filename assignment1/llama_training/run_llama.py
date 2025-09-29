import json
import math
import os
import random
from array import array
from bisect import bisect_right
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cli import parse_args
from classifier import LlamaPretrainingModel
from llama import Llama, load_pretrained
from optimizer import AdamW
from tokenizer import Tokenizer
from utils import (
	finish_wandb,
	get_resume_checkpoint_path,
	init_wandb,
	maybe_resume_from_checkpoint,
	save_model,
)


TQDM_DISABLE = False
# fix the random seed
def seed_everything(seed=11711):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


class WarmupLearningRateScheduler:
	"""Linear warmup scheduler that falls back to a constant learning rate."""

	def __init__(self, base_lr: float, warmup_steps: int):
		self.base_lr = base_lr
		self.warmup_steps = max(0, warmup_steps)

	def lr_at_step(self, step: int) -> float:
		#TODO
		# ====================== Implement lr_at_step here ======================
		# Implement a linear warmup learning rate scheduler.
		#
		# Args:
		#     step (int): Current training step (0-indexed)
		#
		# Returns:
		#     float: Learning rate for the given step
		pass
		# ====================== Implement lr_at_step here ======================

	def __call__(self, step: int) -> float:
		return self.lr_at_step(step)

class PretrainingSequenceDataset(Dataset):
	def __init__(self, data_dir: Path, metadata: dict, block_size: int):
		self.data_dir = data_dir
		self.block_size = block_size
		dtype = metadata.get('dtype', 'uint16')
		if dtype != 'uint16':
			raise ValueError(f"Unsupported dtype {dtype} in metadata")
		self.dtype = np.uint16
		self.entries = []
		self.cumulative = []
		total_sequences = 0
		for file_info in metadata.get('files', []):
			token_file = data_dir / file_info['token_file']
			token_count = int(file_info['token_count'])
			if not token_file.exists() or token_count < self.block_size:
				continue
			memmap = np.memmap(token_file, dtype=self.dtype, mode='r')
			num_seq = token_count // self.block_size
			if num_seq <= 0:
				continue
			usable_tokens = num_seq * self.block_size
			self.entries.append({
				'memmap': memmap,
				'start': 0,
				'end': usable_tokens,
				'token_file': token_file,
			})
			total_sequences += num_seq
			self.cumulative.append(total_sequences)
		self.total_sequences = total_sequences
		if self.total_sequences == 0:
			raise ValueError("No pretraining sequences available. Ensure tokenized data matches block_size.")

	def __len__(self):
		return self.total_sequences

	def __getitem__(self, idx):
		if idx < 0 or idx >= self.total_sequences:
			raise IndexError(idx)
		file_idx = bisect_right(self.cumulative, idx)
		prev = 0 if file_idx == 0 else self.cumulative[file_idx - 1]
		seq_idx = idx if file_idx == 0 else idx - prev
		entry = self.entries[file_idx]
		start_offset = entry['start'] + seq_idx * self.block_size
		end_offset = start_offset + self.block_size
		seq = entry['memmap'][start_offset:end_offset]
		# ensure we have the expected length; fallback to last block if needed
		if seq.shape[0] != self.block_size:
			offset = entry['end'] - self.block_size
			seq = entry['memmap'][offset:offset + self.block_size]
		return torch.tensor(np.array(seq, dtype=np.int64), dtype=torch.long)

	def collate_fn(self, batch):
		token_ids = torch.stack(batch)
		labels = torch.zeros(token_ids.shape[0], dtype=torch.long)
		sents = ['' for _ in batch]
		return {'token_ids': token_ids, 'labels': labels, 'sents': sents}


def tokenize_text_file(input_path: Path, output_path: Path, tokenizer: Tokenizer) -> int:
	token_buffer = array('H')
	with open(input_path, 'r', encoding='utf-8') as fp:
		for line in fp:
			text = line.strip()
			if not text:
				continue
			tokens = tokenizer.encode(text, bos=True, eos=True)
			token_buffer.extend(tokens)
	with open(output_path, 'wb') as out:
		token_buffer.tofile(out)
	token_count = len(token_buffer)
	print(f"Tokenized {input_path} -> {output_path} ({token_count} tokens)")
	return token_count


def preprocess_pretraining_corpus(data_path: str, tokenizer: Tokenizer, tokenized_dir: Optional[str], overwrite: bool = False):
	data_dir = Path(data_path)
	assert data_dir.is_dir(), f"Expected directory for pretraining data, got {data_path}"
	output_dir = Path(tokenized_dir) if tokenized_dir is not None else data_dir / 'tokenized'
	output_dir.mkdir(parents=True, exist_ok=True)
	metadata = {'dtype': 'uint16', 'files': []}
	patterns = ('*.train', '*.val', '*.test', '*.txt', '*.dev')
	input_files = []
	for pattern in patterns:
		input_files.extend(data_dir.glob(pattern))
	seen = set()
	for input_file in sorted(input_files):
		if input_file.name in seen or input_file.is_dir():
			continue
		seen.add(input_file.name)
		output_file = output_dir / f"{input_file.stem}.bin"
		if overwrite or not output_file.exists():
			token_count = tokenize_text_file(input_file, output_file, tokenizer)
		else:
			token_count = output_file.stat().st_size // np.dtype(np.uint16).itemsize
		metadata['files'].append({
			'source': str(input_file),
			'token_file': output_file.name,
			'token_count': int(token_count)
		})
	if not metadata['files']:
		raise ValueError(f"No supported text files found in {data_path}. Expected extensions: {', '.join(patterns)}")
	metadata_path = output_dir / 'metadata.json'
	with open(metadata_path, 'w') as mf:
		json.dump(metadata, mf, indent=2)
	print(f"Wrote pretraining metadata to {metadata_path}")
	return output_dir, metadata

def evaluate_pretraining(dataloader, model, device, marker="val", pad_token_id=None):
	model.eval()
	total_loss = 0.0
	total_tokens = 0
	with torch.no_grad():
		for batch in tqdm(dataloader, desc=marker, disable=TQDM_DISABLE):
			token_ids = batch['token_ids'].to(device)
			logits = model.llama(token_ids, targets=token_ids)[0]
			logits = F.log_softmax(logits, dim=-1)
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = token_ids[..., 1:].contiguous()
			if shift_logits.size(1) == 0:
				continue
			logits_flat = shift_logits.view(-1, shift_logits.size(-1))
			labels_flat = shift_labels.view(-1)
			if pad_token_id is not None:
				valid_mask = labels_flat.ne(pad_token_id)
				if not torch.any(valid_mask):
					continue
				logits_flat = logits_flat[valid_mask]
				labels_flat = labels_flat[valid_mask]
			batch_token_count = labels_flat.numel()
			if batch_token_count == 0:
				continue
			batch_loss = F.nll_loss(logits_flat, labels_flat, reduction='sum')
			total_loss += batch_loss.item()
			total_tokens += batch_token_count
	if total_tokens == 0:
		return float('inf'), float('inf')
	avg_loss = total_loss / total_tokens
	perplexity = math.exp(avg_loss)
	return avg_loss, perplexity


def train(args):
    if args.option != "pretrain":
        raise ValueError("train() only supports the 'pretrain' option.")
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    tokenizer = Tokenizer(None)
    if not args.data_path or not os.path.isdir(args.data_path):
        raise ValueError("Expected --data_path to point to a directory containing pretraining data.")
    train_token_dir, train_metadata = preprocess_pretraining_corpus(
        args.data_path,
        tokenizer,
        args.tokenized_dir,
        args.overwrite_tokenized,
    )
    train_dataset = PretrainingSequenceDataset(train_token_dir, train_metadata, args.block_size)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.micro_batch_size,
        collate_fn=train_dataset.collate_fn,
    )

    dev_dataloader = None
    if args.val_path:
        if not os.path.isdir(args.val_path):
            raise ValueError(f"Expected validation directory, got {args.val_path}")
        val_token_dir, val_metadata = preprocess_pretraining_corpus(
            args.val_path,
            tokenizer,
            args.val_tokenized_dir,
            args.overwrite_tokenized,
        )
        dev_dataset = PretrainingSequenceDataset(val_token_dir, val_metadata, args.block_size)
        dev_dataloader = DataLoader(
            dev_dataset,
            shuffle=False,
            batch_size=args.micro_batch_size,
            collate_fn=dev_dataset.collate_fn,
        )

    test_dataloader = None
    if args.test_path:
        if not os.path.isdir(args.test_path):
            raise ValueError(f"Expected test directory, got {args.test_path}")
        test_token_dir, test_metadata = preprocess_pretraining_corpus(
            args.test_path,
            tokenizer,
            args.test_tokenized_dir,
            args.overwrite_tokenized,
        )
        test_dataset = PretrainingSequenceDataset(test_token_dir, test_metadata, args.block_size)
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=args.micro_batch_size,
            collate_fn=test_dataset.collate_fn,
        )

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        pretrained_model_path=args.pretrained_model_path,
        num_labels=1,
        data_dir='.',
        option=args.option,
    )

    model = LlamaPretrainingModel(config).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    wandb_run = init_wandb(args)
    best_val_loss = float('inf')
    global_step = 0

    micro_batch_size = args.micro_batch_size
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    effective_target = max(1, args.batch_size)
    gradient_accumulation_steps = max(1, math.ceil(effective_target / micro_batch_size))
    effective_batch_size = micro_batch_size * gradient_accumulation_steps
    if effective_batch_size != args.batch_size:
        print(
            f"Using effective batch size {effective_batch_size} (micro {micro_batch_size} x accumulation {gradient_accumulation_steps})"
        )
    updates_per_epoch = (
        math.ceil(len(train_dataloader) / gradient_accumulation_steps) if len(train_dataloader) > 0 else 0
    )
    total_expected_updates = updates_per_epoch * args.epochs if updates_per_epoch > 0 else 0
    base_lr = args.lr
    warmup_steps = max(0, args.warmup_steps)
    if warmup_steps == 0 and args.warmup_ratio > 0 and total_expected_updates > 0:
        warmup_steps = max(1, int(total_expected_updates * args.warmup_ratio))
    if total_expected_updates > 0:
        warmup_steps = min(warmup_steps, total_expected_updates)

    lr_scheduler = WarmupLearningRateScheduler(base_lr, warmup_steps)

    resume_checkpoint_path = None
    resume_state = {}
    resumed_from_checkpoint = False
    resume_state_micro_step = None
    resume_epoch = None
    resume_micro_step = 0
    resume_updates_in_epoch = 0
    resume_applied = True
    start_epoch = 0
    if args.auto_resume:
        resume_checkpoint_path = get_resume_checkpoint_path(args.filepath)
        resume_state = maybe_resume_from_checkpoint(model, optimizer, args, device, resume_checkpoint_path)
        resumed_from_checkpoint = resume_state is not None
        resume_state_micro_step = resume_state.get('micro_step') if resumed_from_checkpoint else None
        resume_applied = not resumed_from_checkpoint
        if resumed_from_checkpoint:
            resume_epoch = int(resume_state.get('epoch') or 0)
            saved_global_step = resume_state.get('global_step')
            if saved_global_step is not None:
                global_step = int(saved_global_step)
            if resume_state_micro_step is not None:
                resume_micro_step = int(resume_state_micro_step)
            resume_updates_in_epoch = int(resume_state.get('updates_in_epoch') or 0)
            best_val_loss = resume_state.get('best_val_loss', best_val_loss)
            if resume_state_micro_step is None:
                if 'micro_step' in resume_state:
                    start_epoch = min(resume_epoch + 1, args.epochs)
                else:
                    start_epoch = min(resume_epoch, args.epochs)
                resume_micro_step = 0
                resume_updates_in_epoch = 0
                resume_applied = True
            else:
                start_epoch = min(resume_epoch, args.epochs)
    if start_epoch >= args.epochs:
        print(f"Loaded checkpoint already completed {args.epochs} epochs; skipping training loop.")

    epoch_iter = range(start_epoch, args.epochs)
    initial_lr = lr_scheduler(global_step)
    for group in optimizer.param_groups:
        group['lr'] = initial_lr
    optimizer.zero_grad()

    for epoch in tqdm(epoch_iter):
        model.train()
        train_loss = 0.0
        num_batches = 0
        token_loss_total = 0.0
        token_count_total = 0
        steps_since_update = 0
        accumulated_token_loss = 0.0
        accumulated_token_count = 0
        apply_resume_skips = (
            not resume_applied
            and args.auto_resume
            and resumed_from_checkpoint
            and resume_epoch is not None
            and epoch == resume_epoch
            and resume_state_micro_step is not None
        )

        total_micro_steps = len(train_dataloader)
        gradient_updates_this_epoch = math.ceil(total_micro_steps / gradient_accumulation_steps)
        if apply_resume_skips:
            skip_micro_steps = min(max(resume_micro_step, 0), total_micro_steps)
            initial_updates = min(resume_updates_in_epoch, gradient_updates_this_epoch)
        else:
            skip_micro_steps = 0
            initial_updates = 0
        updates_completed_in_epoch = initial_updates
        pbar = tqdm(
            total=gradient_updates_this_epoch,
            desc=f'train-{epoch}',
            disable=TQDM_DISABLE,
            initial=initial_updates,
        )
        for step, batch in enumerate(train_dataloader):
            if apply_resume_skips and step < skip_micro_steps:
                continue
            b_ids = batch['token_ids'].to(device)

            logits = model.llama(b_ids, targets=b_ids)[0]
            logits = F.log_softmax(logits, dim=-1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = b_ids[..., 1:].contiguous()
            token_count = shift_labels.numel()

            if shift_logits.size(1) > 0:
                token_loss_sum = F.nll_loss(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum',
                )
                loss = token_loss_sum / gradient_accumulation_steps / token_count
                token_loss_total += token_loss_sum.item()
                token_count_total += token_count
                accumulated_token_loss += token_loss_sum.item()
                accumulated_token_count += token_count
            else:
                loss = torch.tensor(0.0, device=b_ids.device, requires_grad=True)
                token_loss_sum = loss

            loss.backward()
            steps_since_update += 1

            update_now = (steps_since_update == gradient_accumulation_steps) or (step == total_micro_steps - 1)

            if update_now:
                lr_this_step = lr_scheduler(global_step + 1)
                for group in optimizer.param_groups:
                    group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                steps_since_update = 0
                global_step += 1
                updates_completed_in_epoch += 1
                pbar.update(1)

                denom = accumulated_token_count if accumulated_token_count > 0 else 1
                step_token_loss = accumulated_token_loss / denom
                train_loss += step_token_loss
                num_batches += 1
                accumulated_token_loss = 0.0
                accumulated_token_count = 0
                avg_token_loss = train_loss / num_batches
                pbar.set_postfix(
                    {
                        'step_tok_loss': f'{step_token_loss:.4f}',
                        'avg_tok_loss': f'{avg_token_loss:.4f}',
                    }
                )
                if wandb_run:
                    current_lr = optimizer.param_groups[0]['lr']
                    wandb_run.log(
                        {
                            'train/token_loss_step': step_token_loss,
                            'train/token_loss_avg': avg_token_loss,
                            'lr': current_lr,
                        },
                        step=global_step,
                    )

                should_validate = (
                    args.val_per_steps > 0
                    and dev_dataloader is not None
                    and global_step % args.val_per_steps == 0
                )
                if should_validate:
                    val_loss, val_ppl = evaluate_pretraining(dev_dataloader, model, device)
                    print(
                        f"step {global_step}: val loss (per token) :: {val_loss :.4f}, perplexity :: {val_ppl :.4f}"
                    )
                    if wandb_run:
                        wandb_run.log({'val/token_loss': val_loss, 'val/perplexity': val_ppl}, step=global_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_model(
                            model,
                            optimizer,
                            args,
                            config,
                            args.filepath,
                            epoch=epoch,
                            global_step=global_step,
                            micro_step=step + 1,
                            updates_in_epoch=updates_completed_in_epoch,
                            best_dev_acc=None,
                            best_val_loss=best_val_loss,
                        )
                    model.train()
                    if args.auto_resume and resume_checkpoint_path is not None:
                        save_model(
                            model,
                            optimizer,
                            args,
                            config,
                            resume_checkpoint_path,
                            epoch=epoch,
                            global_step=global_step,
                            micro_step=step + 1,
                            updates_in_epoch=updates_completed_in_epoch,
                            best_dev_acc=None,
                            best_val_loss=best_val_loss,
                            quiet=True,
                        )

        pbar.close()
        train_loss = train_loss / num_batches if num_batches > 0 else 0.0

        avg_token_loss = (token_loss_total / token_count_total) if token_count_total > 0 else float('inf')
        perplexity = math.exp(avg_token_loss) if avg_token_loss != float('inf') else float('inf')
        print(f"epoch {epoch}: train loss/token :: {avg_token_loss :.4f}, perplexity :: {perplexity :.4f}")
        if wandb_run:
            wandb_run.log(
                {
                    'train/token_loss_epoch': avg_token_loss,
                    'train/perplexity': perplexity,
                    'lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                },
                step=global_step,
            )
        save_model(
            model,
            optimizer,
            args,
            config,
            args.filepath,
            epoch=epoch,
            global_step=global_step,
            micro_step=None,
            updates_in_epoch=updates_completed_in_epoch,
            best_dev_acc=None,
            best_val_loss=best_val_loss,
        )

        if args.auto_resume and resume_checkpoint_path is not None:
            save_model(
                model,
                optimizer,
                args,
                config,
                resume_checkpoint_path,
                epoch=epoch,
                global_step=global_step,
                micro_step=None,
                updates_in_epoch=updates_completed_in_epoch,
                best_dev_acc=None,
                best_val_loss=best_val_loss,
                quiet=True,
            )
        if args.auto_resume and not resume_applied:
            resume_applied = True

    if test_dataloader is not None:
        test_loss, test_ppl = evaluate_pretraining(test_dataloader, model, device, 'test')
        print(f"final test loss (per token) :: {test_loss :.4f}, perplexity :: {test_ppl :.4f}")
        if wandb_run:
            wandb_run.log({'test/token_loss': test_loss, 'test/perplexity': test_ppl}, step=global_step)

def generate_sentence(args, prefix, outfile, max_new_tokens = 75, temperature = 0.0, top_k = 20):
	with torch.no_grad():
		device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
		ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float32) if args.use_gpu else nullcontext()
		llama = load_pretrained(args.pretrained_model_path)
		llama = llama.to(device)
		print(f"load model from {args.pretrained_model_path}")
		enc = Tokenizer(args.max_sentence_len)

		start_ids = enc.encode(prefix, bos=True, eos=False)
		x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

		# run generation
		with torch.no_grad():
			with ctx:
				y = llama.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
				sentence = enc.decode(y[0].tolist())
				print(f"Temperature is {temperature}")
				print(sentence)
				print('---------------')
				writer = open(outfile, 'w')
				writer.write(sentence)
				print(f"Wrote generated sentence to {outfile}.")
				writer.close()



if __name__ == "__main__":
	args = parse_args()
	run_name = args.run_name or "run"
	args.filepath = f"{run_name}-{args.option}-{args.epochs}-{args.lr}.pt"
	seed_everything(args.seed)
	try:
		if args.option == "generate":
			prefix = "White Bird is a 2023 American war drama movie starring"
			generate_sentence(args, prefix, args.generated_sentence_low_temp_out, max_new_tokens=75, temperature=0.0, top_k = 20)
			generate_sentence(args, prefix, args.generated_sentence_high_temp_out, max_new_tokens=75, temperature=0.5, top_k = 50)
		elif args.option == "pretrain":
			train(args)
		else:
			raise ValueError(f"Invalid option: {args.option}")
	finally:
		finish_wandb()
