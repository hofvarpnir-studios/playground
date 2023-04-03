import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict

from playground.tracker.tracker import Tracker
from pathlib import Path

TOP_LEVEL = Path(__file__).parent
CONFIGS = TOP_LEVEL / "configs"
DATA = TOP_LEVEL / "data"


class Trainer:
    def __init__(
        self,
        model_id: str,
        dataset_path: str,
        batch_size: int,
        max_length: int,
        epochs: int,
        lr: float,
        tracker: Tracker,
        checkpoint_name: str,
    ):
        ### Init

        self.epochs = epochs
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.tracker = tracker
        self.checkpoint_name = checkpoint_name

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
                "train": str(DATA / dataset_path),
            },
            cache_dir="./cache",
        )

        with self.accelerator.main_process_first():
            self.processed_dataset = dataset.map(
                self.prepare_data,
                load_from_cache_file=False,
                desc="Formatting data...",
                remove_columns=["instruction", "input", "output"],
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
    def format_prompt(example: dict[str, str]) -> str:
        if example["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the requested task.

Instruction: {example["instruction"]}

Input: {example["input"]}

Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the requested task.
                
Instruction: {example["instruction"]}

Response:"""

    def prepare_data(self, example: dict[str, str]):
        input = self.format_prompt(example)
        model_inputs = self.tokenizer(
            input,
            text_target=example["output"],
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return model_inputs

    def train(self):
        for _ in range(self.epochs):
            self.model.train()
            k = 0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.tracker.log_iteration_time(batch.size(0), k)
                if step % 1000 == 0:
                    print("loss: ", loss.detach().float())
                    self.tracker.add_scalar(
                        "metrics/train-loss", loss.detach().float(), k
                    )
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        self.accelerator.save(
                            get_peft_model_state_dict(
                                self.model,
                                state_dict=self.accelerator.get_state_dict(self.model),
                            ),
                            self.checkpoint_name,
                        )
                k += 1
        self.accelerator.wait_for_everyone()
