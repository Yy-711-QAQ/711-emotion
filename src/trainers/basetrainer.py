import os
import torch


class BaseTrainer:
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.dataloaders = dataloaders

        self.saving_path = "./savings"
        os.makedirs(self.saving_path, exist_ok=True)
        os.makedirs(os.path.join(self.saving_path, "models"), exist_ok=True)

        self.best_model = None
        self.best_epoch = -1

    def get_saving_file_name(self):
        dataset = self.args.get("dataset", "dataset")
        model = self.args.get("model", "model")
        mod = self.args.get("modalities", "mod")
        epoch = self.best_epoch if self.best_epoch != -1 else 0
        return f"{dataset}_{model}_{mod}_best_epoch{epoch}.pt"

    def save_model(self):
        if self.best_model is None:
            print("No best model to save.")
            return
        save_dir = os.path.join(self.saving_path, "models")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, self.get_saving_file_name())
        torch.save(self.best_model, save_path)
        print(f"Saved best model to: {save_path}")
