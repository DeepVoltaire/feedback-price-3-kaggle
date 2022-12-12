import torch
from datetime import datetime
import time
import os
import numpy as np

from .model import FeedbackModel


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    # print(f"max len is {mask_len}")
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


def train(train_dataset, validation_dataset, TrainGlobalConfig, device, net, fold_nb):
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=TrainGlobalConfig.num_workers,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=TrainGlobalConfig.num_workers,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig, fold_nb=fold_nb)
    fitter.fit(train_loader, validation_loader)


def MCRMSE(y_trues, y_preds):
    """Mean Columnwise root mean squared error"""
    scores = []
    class_indices = y_trues.shape[1]
    for i in range(class_indices):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = ((y_true - y_pred) ** 2).mean() ** 0.5  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores


class Fitter:
    def __init__(self, model, device, config, fold_nb):

        self.config = config
        self.epoch = 0
        self.model = model
        self.device = device
        self.best_metric = 1000
        self.early_stopping_counter = 0
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.criterion = config.criterion

        self.log_folder = f"trained_models/{config.exp_name}/fold{fold_nb}"
        os.makedirs(self.log_folder, exist_ok=True)
        self.log_path = f"{self.log_folder}/log.log"
        self.log(f"Fitter prepared. Device is {self.device}, logging to {self.log_path}")

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            lr = self.optimizer.param_groups[0]["lr"]

            t = time.perf_counter()
            losses = self.train_one_epoch(train_loader)

            self.log(
                f"[RESULT]: Train. Epoch: {self.epoch}, loss: {losses.avg:.5f}, time: {(time.perf_counter() - t):.5f}"
            )

            t = time.perf_counter()
            losses, preds = self.validation(validation_loader)
            score, scores = get_score(validation_loader.dataset.targets, preds)

            self.log(
                f"[RESULT]: Validation. Epoch: {self.epoch}, loss: {losses.avg:.5f}, Score: {score}, Scores {scores}, time: {(time.perf_counter() - t):.5f}"
            )

            if losses.avg < self.best_metric:
                self.best_metric = losses.avg
                self.save("best-checkpoint.bin", score)
                self.early_stopping_counter = 0

            if self.epoch > 5:
                # warm up epochs without reducing LR
                self.scheduler.step(metrics=losses.avg)
                if lr < 1e-6:  # only early stop when we have tried reducing the LR multiple times
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter > self.config.early_stopping_patience:
                        self.log("Early Stopping")
                        return

            self.epoch += 1

    def train_one_epoch(self, train_loader):
        self.model.train()

        losses = AverageMeter()
        t = time.perf_counter()
        for step, (inputs, targets) in enumerate(train_loader):
            if self.config.verbose:
                if step > 0 and step % self.config.verbose_step == 0:
                    self.log(f"Train Step {step}, loss: {losses.avg:.5f}, time: {(time.perf_counter() - t):.5f}")
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            targets = targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            batch_size = targets.size(0)
            losses.update(loss.detach().item(), batch_size)

            # loss.backward()
            # self.optimizer.step()

        self.model.eval()
        return losses

    def validation(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        preds = []

        t = time.perf_counter()
        for step, (inputs, targets) in enumerate(val_loader):
            with torch.no_grad():
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                targets = targets.to(self.device)

                batch_size = targets.size(0)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                losses.update(loss.detach().item(), batch_size)
                preds.append(outputs.cpu().numpy())

        return losses, np.concatenate(preds)

    def save(self, path, loss_value):
        torch.save(
            {"state_dict": self.model.state_dict(), "loss": loss_value, "optimizer": self.optimizer.state_dict()},
            f"{self.log_folder}/{path}",
        )

    def log(self, message):
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open(self.log_path, "a+") as file:
            file.write(f"{date}: {message} \n")
        if self.config.verbose:
            print(f"{date}: {message}")


def load_models(backbone, p_dropout, fold_nbs, config, device):
    models = []
    for fold_nb in fold_nbs:
        model = FeedbackModel(backbone=backbone, p_dropout=p_dropout)
        model_path = f"trained_models/{config.exp_name}/fold{fold_nb}/best-checkpoint.bin"
        checkpoint = torch.load(model_path)
        print(f"Loading {model_path}")
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)
    return models


def testing(device, test_dataset, config, models=None, fold_nbs=None, backbone=None, p_dropout=None, submit=False):
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=config.num_workers,
    )

    if models is None:
        models = load_models(backbone, p_dropout, fold_nbs, config, device)
    t = time.perf_counter()
    y_pred = []
    for step, data in enumerate(test_loader):
        with torch.no_grad():
            if submit:
                inputs = collate(data)
            else:
                inputs, targets = data
                inputs = collate(inputs)
                targets = targets.to(device)
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            outputs = models[0](inputs)
            for model_nb in range(1, len(models)):
                outputs += models[model_nb](inputs)
            outputs /= len(models)

            y_pred.append(outputs.cpu().numpy())

    y_pred = np.concatenate(y_pred)
    if not submit:
        mean_score, scores = get_score(test_loader.dataset.targets, y_pred)
        # print(
        #     f"[RESULT]: Testset. Score: {mean_score:.3f}, Scores {np.round(scores, 3)}, time: {(time.perf_counter() - t):.5f}"
        # )
        return y_pred, mean_score, scores
    return y_pred


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
