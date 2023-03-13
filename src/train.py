import torch
import torch.nn as nn

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

        self.init_data_loaders()
        self.init_optimizer()
        self.init_scheduler()
        self.model = Discriminator(data.dim, params)
        self.loss = nn.BCEWithLogitsLoss()
        self.bayesian = params.get("bayesian", False)
        if self.bayesian:
            self.losses = {
                "lr": [],
                "train_loss": [],
                "train_bce_loss": [],
                "train_kl_loss": [],
                "test_loss": [],
                "test_bce_loss": [],
                "test_kl_loss": []
            }
        else:
            self.losses = {
                "lr": [],
                "train_loss": [],
                "test_loss": []
            }


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
                self.optimizer,
                step_size = self.params["lr_decay_epochs"],
                gamma = self.params["lr_decay_factor"],
            )
        elif self.scheduler_type == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                self.params.get("max_lr", self.params["lr"]*10),
                epochs = self.params["epochs"],
                steps_per_epoch=self.train_batches
            )
        else:
            raise ValueError(f"Unknown LR scheduler '{self.scheduler_type}'")


    def batch_loss(self, x_true, x_fake):
        y_true = self.model(x_true)
        y_fake = self.model(x_fake)
        loss_true = self.loss(y_true, torch.ones_like(y_true))
        loss_fake = self.loss(y_fake, torch.zeros_like(y_fake))
        bce_loss = 0.5 * (loss_true + loss_fake)
        if self.bayesian:
            kl_loss = sum(
                layer.KL() for layer in self.model.bayesian_layers
            ) / self.train_samples
        else:
            kl_loss = 0
        loss = bce_loss + kl_loss
        return loss, bce_loss, kl_loss


    def train(self):
        for epoch in range(self.params["epochs"]):
            self.model.train()
            epoch_losses, epoch_bce_losses, epoch_kl_losses = [], [], []
            for x_true, x_fake in zip(self.train_loader_true, self.train_loader_fake):
                self.optimizer.zero_grad()
                loss, bce_loss, kl_loss = self.batch_loss(x_true, x_fake)
                epoch_losses.append(loss.detach())
                epoch_bce_losses.append(bce_loss.detach())
                epoch_kl_losses.append(kl_loss.detach())
                loss.backward()
                self.optimizer.step()
                if self.scheduler_type == "one_cycle":
                    self.scheduler.step()
            if self.scheduler_type == "step":
                self.scheduler.step()

            test_loss, test_bce_loss, test_kl_loss = self.test_loss()
            train_loss = torch.stack(epoch_losses).mean()
            self.losses["train_loss"].append(train_loss)
            self.losses["test_loss"].append(test_loss)
            self.losses["lr"].append(self.optimizer.param_groups[0]["lr"])
            if self.bayesian:
                self.losses["train_bce_loss"].append(torch.stack(epoch_bce_losses).mean())
                self.losses["train_kl_loss"].append(torch.stack(epoch_kl_losses).mean())
                self.losses["test_bce_loss"].append(test_bce_loss)
                self.losses["test_kl_loss"].append(test_kl_loss)
                print(f"    Epoch {epoch:3d}: train loss {train_loss:.6f}, " +
                      f"test loss {test_loss:.6f}")


    def test_loss(self):
        self.model.eval()
        with torch.no_grad():
            losses = []
            bce_losses = []
            kl_losses = []
            for x_true, x_fake in zip(self.test_loader_true, self.test_loader_fake):
                loss, bce_loss, kl_loss = self.batch_loss(x_true, x_fake)
                losses.append(loss)
                bce_losses.append(bce_loss)
                kl_losses.append(kl_loss)
            return (
                torch.stack(losses).mean(),
                torch.stack(bce_losses).mean(),
                torch.stack(kl_losses).mean()
            )


    def predict(self):
        self.model.eval()
        with torch.no_grad():
            y_true = torch.cat([
                self.model(x_true).sigmoid()
                for x_true in self.val_loader_true
            ])
            y_fake = torch.cat([
                self.model(x_fake).sigmoid()
                for x_fake in self.val_loader_fake
            ])
            w_true = (1 - y_true) / y_true
            w_fake = y_fake / (1 - y_fake)
            return w_true, w_fake


    def save(self, file: str):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "losses": self.losses
        }, file)


    def load(self, file: str):
        state_dicts = torch.load(file, map_location=self.device)
        self.optimizer.load_state_dict(state_dicts["optimizer"])
        self.model.load_state_dict(state_dicts["model"])
        self.losses = state_dicts["model"]
