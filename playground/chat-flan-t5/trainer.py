import torch
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator
import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq


from playground.tracker.tracker import Tracker

class Trainer:
    def __init__(
        self,
        model_id: str = "google/flan-t5-xxl",
        dataset_id: str = "Anthropic/hh-rlhf",
        epochs: int = 1,
        lr: float = 1e-4,
    ):
        self.accelerator = Accelerator()
        self.epochs = epochs
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.tracker = tracker

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(self.accelerator.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * self.epochs),
        )

        self.model, self.train_dataloader, self.eval_dataloader, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
        )
        self.model = torch.compile(self.model)
        self.accelerator.state.deepspeed_plugin.zero_stage == 3

    
    def prepare_data(self):
        # Load dataset from the hub
        dataset = load_dataset(self.dataset_id)
        # Load tokenizer of FLAN-t5-base

        print(f"Train dataset size: {len(dataset['train'])}")
        print(f"Test dataset size: {len(dataset['test'])}")
    
    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                if step%1000 == 0:
                    print("loss: ",loss.detach().float())
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        self.accelerator.save(
                            get_peft_model_state_dict(self.model, state_dict=self.accelerator.get_state_dict(self.model)), checkpoint_name
                        )
            self.eval(epoch)
    
    def eval(self, epoch: int, total_loss: float) -> tuple[float, list[str]]:
        self.model.eval()
        eval_loss = 0
        eval_preds = []
        for _, batch in enumerate(tqdm(self.eval_dataloader)):
            with torch.no_grad():
                outputs = self.model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            preds = self.accelerator.gather_for_metrics(torch.argmax(outputs.logits, -1)).detach().cpu().numpy()
            eval_preds.extend(self.tokenizer.batch_decode(preds, skip_special_tokens=True))
        eval_epoch_loss = eval_loss / len(self.train_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(self.eval_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        self.accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        self.accelerator.wait_for_everyone()
        self.accelerator.save(
            get_peft_model_state_dict(self.model, state_dict=self.accelerator.get_state_dict(self.model)), checkpoint_name
        )
        self.accelerator.wait_for_everyone()
        return eval_epoch_loss, eval_preds