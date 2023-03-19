import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict

from playground.tracker.tracker import ConsoleTracker, Tracker


class Trainer:
    def __init__(
        self,
        model_id: str = "google/flan-t5-xxl",
        dataset_path: str = "./alpaca_data.json",
        batch_size: int = 4,
        max_length: int = 512,
        epochs: int = 1,
        lr: float = 1e-4,
        tracker: Tracker = ConsoleTracker(),
    ):
        ### Init

        self.epochs = epochs
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.tracker = tracker
        self.checkpoint_name = "chaT5_lora.pt"

        ### Load model

        self.accelerator = Accelerator()

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        model = get_peft_model(model, peft_config)

        ### Load dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dataset = load_dataset(
            "json",
            data_files={
                "train": dataset_path,
            },
            cache_dir="./cache",
        )

        with self.accelerator.main_process_first():
            self.processed_dataset = dataset.map(
                self.prepare_data, batched=True, batch_size=batch_size
            )["train"]

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)

        train_dataloader = DataLoader(
            self.processed_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

        ### Init optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * self.epochs),
        )

        ### Prepare model

        (
            self.model,
            self.train_dataloader,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(model, train_dataloader, optimizer, lr_scheduler)
        self.model = torch.compile(self.model)

    @staticmethod
    def format_prompt(instruction: str, input: str) -> str:
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the requested task.

Instruction: {instruction}

Input: {input}

Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the requested task.
                
Instruction: {instruction}

Response:"""

    def prepare_data(self, examples: list[dict[str, str]]):
        inputs = [
            self.format_prompt(example["instruction"], example["input"])
            for example in examples
        ]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [example["output"] for example in examples],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self):
        for _ in range(self.epochs):
            self.model.train()
            total_loss = 0
            k=0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.tracker.log_iteration_time(batch.size(0), k)
                if step % 1000 == 0:
                    print("loss: ", loss.detach().float())
                    self.tracker.add_scalar("metrics/train-loss", loss.detach().float(), k)
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        self.accelerator.save(
                            get_peft_model_state_dict(
                                self.model,
                                state_dict=self.accelerator.get_state_dict(self.model),
                            ),
                            self.checkpoint_name,
                        )
                k+=1
