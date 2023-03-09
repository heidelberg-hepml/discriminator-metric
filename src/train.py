import torch

from .model import Discriminator
from .dataset import DiscriminatorData

class DiscriminatorTraining:
    def __init__(
        self,
        params: dict,
        device: torch.device,
        data: DiscriminatorData
    ):
        self.params = params
        self.device = device
        self.data = data

        self.losses = {}
        self.init_data_loaders()
        self.init_optimizer()
        self.init_scheduler()
        self.model = Discriminator(data.dim, params)

    def init_data_loaders(self):
        make_loader = lambda dataset, mode: torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = self.params["batch_size"],
            shuffle = mode == "train",
            drop_last = mode in ["train", "val"]
        )
        self.train_loader_true = make_loader(self.data.train_true, "train")
        self.train_loader_fake = make_loader(self.data.train_fake, "train")
        self.test_loader_true = make_loader(self.data.test_true, "test")
        self.test_loader_fake = make_loader(self.data.test_fake, "test")
        self.val_loader_true = make_loader(self.data.val_true, "val")
        self.val_loader_fake = make_loader(self.data.val_fake, "val")

        self.train_batches = min(len(self.train_loader_true), len(self.train_loader_fake))
        self.train_samples = len(self.data.train_true) + len(self.data.train_fake)

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.params.get("lr", 1e-4),
            betas = self.params.get("betas", [0.9, 0.999]),
            eps = self.params.get("eps", 1e-6),
            weight_decay = self.params.get("weight_decay", 0.)
        )

    def init_scheduler(self):
        self.scheduler_type = self.params.get("lr_scheduler", "one_cycle")
        if self.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size = self.params["lr_decay_epochs"],
                gamma = self.params["lr_decay_factor"],
            )
        elif self.scheduler_type == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                self.params.get("max_lr", self.params["lr"]*10),
                epochs = self.params["epochs"],
                steps_per_epoch=self.train_batches
            )
        else:
            raise ValueError(f"Unknown LR scheduler '{self.scheduler_type}'")

    def batch_loss(self, x_true, x_fake):
        pass

    def train(self):
        self.model.train()
        for epoch in range(self.params["epochs"]):
            for x_true, x_fake in zip(self.train_loader_true, self.train_loader_fake):
                self.optimizer.zero_grad()
                bce_loss = self.batch_loss(x_true, x_fake)
                if self.bayesian:
                    kl_loss = sum(
                        layer.KL() for layer in self.model.bayesian_layers
                    ) / self.train_samples
                    loss = bce_loss + kl_loss
                else:
                    loss = bce_loss
                loss.backward()
                self.optimizer.step()
                if self.scheduler_type == "one_cycle":
                    self.scheduler.step()
            if self.scheduler_type == "step":
                self.scheduler.step()

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for x_true in self.val_loader_true:
                pass
            for x_fake in self.val_loader_fake:
                pass
        return y_true, y_fake

    def save(self, file):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "losses": self.losses
        }, file)

    def load(self):
        state_dicts = torch.load(file, map_location=self.device)
        self.optimizer.load_state_dict(state_dicts["optimizer"])
        self.model.load_state_dict(state_dicts["model"])
        self.losses = state_dicts["model"]
