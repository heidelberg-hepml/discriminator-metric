import torch
import torch.nn as nn
import numpy as np

from .model import Discriminator
from .dataset import DiscriminatorData

class DiscriminatorTraining:
    """
    Class to build, train, evaluate, save and load discriminator models for generative
    networks.
    """

    def __init__(
        self,
        params: dict,
        device: torch.device,
        data: DiscriminatorData
    ):
        """
        Build the network and data loaders.

        Args:
            params: Dict with architecture and training hyperparameters
            device: Pytorch device used for training and evaluation
            data: DiscriminatorData object containing the training and evaluation data
        """
        self.params = params
        self.device = device
        self.data = data

        self.init_data_loaders()
        self.model = Discriminator(data.dim, params)
        self.model.to(self.device)
        self.init_optimizer()
        self.init_scheduler()
        self.loss = nn.BCEWithLogitsLoss()
        self.bayesian = params.get("bayesian", False)
        if self.bayesian:
            self.losses = {
                "lr": [],
                "train_loss": [],
                "train_bce_loss": [],
                "train_kl_loss": [],
                "val_loss": [],
                "val_bce_loss": [],
                "val_kl_loss": []
            }
        else:
            self.losses = {
                "lr": [],
                "train_loss": [],
                "val_loss": []
            }


    def init_data_loaders(self):
        """
        Initializes the data loaders using the DiscriminatorData object given to __init__.
        """
        make_loader = lambda data, mode: torch.utils.data.DataLoader(
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(data, device=self.device)
            ),
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
        """
        Initialized the Adam optimizer.
        """
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.params.get("lr", 1e-4),
            betas = self.params.get("betas", [0.9, 0.999]),
            eps = self.params.get("eps", 1e-6),
            weight_decay = self.params.get("weight_decay", 0.)
        )


    def init_scheduler(self):
        """
        Initialized the LR scheduler. Currently, one-cycle and step schedulers are supported.
        """
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


    def batch_loss(self, x_true, x_fake) -> tuple[torch.Tensor, ...]:
        """
        Computes the loss function for a single batch.

        Args:
            x_true: Discriminator output for truth samples
            x_fake: Discriminator output for generated samples

        Returns:
            loss: total loss
            bce_loss: binary cross entropy loss
            kl_loss: KL loss term (if network is bayesian, otherwise 0)
        """
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
            kl_loss = torch.tensor(0.)
        loss = bce_loss + kl_loss
        return loss, bce_loss, kl_loss


    def train(self):
        """
        Main training loop
        """
        for epoch in range(self.params["epochs"]):
            self.model.train()
            epoch_losses, epoch_bce_losses, epoch_kl_losses = [], [], []
            for (x_true, ), (x_fake, ) in zip(self.train_loader_true, self.train_loader_fake):
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

            val_loss, val_bce_loss, val_kl_loss = self.val_loss()
            train_loss = torch.stack(epoch_losses).mean()
            self.losses["train_loss"].append(train_loss.item())
            self.losses["val_loss"].append(val_loss.item())
            self.losses["lr"].append(self.optimizer.param_groups[0]["lr"])
            if self.bayesian:
                self.losses["train_bce_loss"].append(torch.stack(epoch_bce_losses).mean().item())
                self.losses["train_kl_loss"].append(torch.stack(epoch_kl_losses).mean().item())
                self.losses["val_bce_loss"].append(val_bce_loss.item())
                self.losses["val_kl_loss"].append(val_kl_loss.item())
            print(f"    Epoch {epoch:3d}: train loss {train_loss:.6f}, " +
                  f"val loss {val_loss:.6f}")


    def val_loss(self) -> tuple[torch.Tensor, ...]:
        """
        Computes the loss for the validation data set

        Returns:
            loss: total loss
            bce_loss: binary cross entropy loss
            kl_loss: KL loss term (if network is bayesian, otherwise 0)
        """
        self.model.eval()
        with torch.no_grad():
            losses = []
            bce_losses = []
            kl_losses = []
            for (x_true, ), (x_fake, ) in zip(self.val_loader_true, self.val_loader_fake):
                loss, bce_loss, kl_loss = self.batch_loss(x_true, x_fake)
                losses.append(loss)
                bce_losses.append(bce_loss)
                kl_losses.append(kl_loss)
            return (
                torch.stack(losses).mean(),
                torch.stack(bce_losses).mean(),
                torch.stack(kl_losses).mean()
            )


    def predict(self) -> tuple[np.ndarray, ...]:
        """
        Computes the classifier weights and loss for the test dataset. If the network is
        Bayesian, the last dimension of the returned numpy arrays corresponds to the
        samples from the trainable weights posterior.

        Returns:
            w_true: Classifier weights for truth samples from the test dataset
            w_fake: Classifier weights for generated samples from the test dataset
            clf_score: Classifier score for the test dataset
        """
        if self.bayesian:
            w_true_all, w_fake_all, clf_score_all = [], [], []
            for i in range(self.params["bayesian_samples"]):
                for layer in self.model.bayesian_layers:
                    layer.reset_random()
                w_true, w_fake, clf_score = self.predict_single()
                w_true_all.append(w_true)
                w_fake_all.append(w_fake)
                clf_score_all.append(clf_score)
            return (
                np.stack(w_true_all, axis=1),
                np.stack(w_fake_all, axis=1),
                np.stack(clf_score_all, axis=0)
            )
        else:
            return self.predict_single()


    def predict_single(self):
        """
        Computes the classifier weights and loss for the test dataset.

        Returns:
            w_true: Classifier weights for truth samples from the test dataset
            w_fake: Classifier weights for generated samples from the test dataset
            clf_score: Classifier score for the test dataset
        """
        self.model.eval()
        with torch.no_grad():
            y_true = torch.cat([
                self.model(x_true) for (x_true, ) in self.test_loader_true
            ])
            y_fake = torch.cat([
                self.model(x_fake) for (x_fake, ) in self.test_loader_fake
            ])
            y_true_sig = y_true.sigmoid().flatten()
            y_fake_sig = y_fake.sigmoid().flatten()
            w_true = (1 - y_true_sig) / y_true_sig
            w_fake = y_fake_sig / (1 - y_fake_sig)
            min_size = min(len(y_true), len(y_fake))
            clf_score = self.loss(
                torch.cat((y_true[:min_size], y_fake[:min_size])),
                torch.cat((
                    torch.ones_like(y_true[:min_size]),
                    torch.zeros_like(y_true[:min_size])
                ))
            )
            return w_true.cpu().numpy(), w_fake.cpu().numpy(), clf_score.cpu().numpy()


    def save(self, file: str):
        """
        Saves the model, optimizer and losses.

        Args:
            file: Output file name
        """
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "losses": self.losses
        }, file)


    def load(self, file: str):
        """
        Loads the model, optimizer and losses.

        Args:
            file: Input file name
        """
        state_dicts = torch.load(file, map_location=self.device)
        self.optimizer.load_state_dict(state_dicts["optimizer"])
        self.model.load_state_dict(state_dicts["model"])
        self.losses = state_dicts["losses"]
