# mnist_full_test.py
"""
Example script demonstrating the full capabilities of the Logger.

This script trains a simple CNN on MNIST and demonstrates:
- Console and File logging
- Weights & Biases (W&B) integration
- TensorBoard integration
- Model architecture logging
- Token/Parameter estimation
- Metrics logging (histograms, scalars)
"""
import sys
import os
from pathlib import Path

# Dodaj katalog nadrzędny do ścieżki
current_dir = Path(__file__).parent.absolute()
root_dir = current_dir.parent
sys.path.append(str(root_dir))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from logger.logger import MetricsLogger
import logging

class Net(nn.Module):
    """Simple CNN for MNIST classification.

    Architecture:
        Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d -> Dropout -> Linear -> ReLU -> Dropout -> Linear
    """
    def __init__(self):
        """Initialize the network with two conv layers and two fc layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward pass of the neural network.

        Args:
            x: Input image tensor (N, 1, 28, 28).

        Returns:
            Log-softmax probabilities (N, 10).
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def full_feature_test():
    """Run a full feature test of the logger using a training loop."""
    log_dir = current_dir / "full_test_logs"
    
    # Włącz wszystko co się da!
    logger = MetricsLogger(
        name="MNIST_FullTest",
        log_dir=str(log_dir),
        level=logging.DEBUG,                 # żeby zobaczyć też debug()
        log_to_console=True,
        log_to_file=True,
        use_tensorboard=True,                # działa od razu
        use_wandb=False,                     # ustaw na True jeśli masz wandb login
        # wandb_project="mnist-test",       # odkomentuj jeśli używasz W&B
        # wandb_entity="twoje_konto",
    )

    # 1. Podstawowe funkcje
    logger.debug("To jest wiadomość DEBUG")
    logger.info("Start pełnego testu loggera!")
    logger.warning("To tylko test – nie panikuj")
    logger.separate("~", 80)
    logger.header("PEŁNY TEST LOGGERA", char="█", length=80)

    # 2. Konfiguracja
    config = {
        "batch_size": 64,
        "epochs": 2,
        "lr": 1.0,
        "gamma": 0.7,
        "seed": 42,
        "model": "SimpleCNN",
        "notes": "Test wszystkich funkcji loggera"
    }
    logger.log_config(config, title="Hyperparameters & Config")

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Urządzenie: {device}")

    # 3. Dane
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    with logger.timer("Pobieranie i przygotowanie danych"):
        dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST('./data', train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000)

    # 4. Model
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config["gamma"])

    dummy_input = torch.randn(1, 1, 28, 28).to(device)

    # Szczegółowa analiza architektury (z kształtami!)
    logger.log_model_info(model, input_size=dummy_input)

    # Wykres modelu (torchviz)
    logger.log_model_graph(model, dummy_input, filename="mnist_cnn_graph")

    # Pamięć GPU (jeśli jest)
    logger.log_gpu_memory(gpu_id=0)

    # 5. Trening
    global_step = 0
    best_val_acc = 0.0

    for epoch in range(1, config["epochs"] + 1):
        logger.log_epoch_start(epoch, config["epochs"])

        model.train()
        epoch_losses = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

            # Logowanie kroku + metryki do TB/W&B
            if batch_idx % 50 == 0:
                logger.log_step(
                    step=batch_idx,
                    total_steps=len(train_loader),
                    metrics={"loss": loss_val, "lr": scheduler.get_last_lr()[0]},
                    log_interval=50
                )
                logger.log_metrics(
                    {"loss": loss_val, "lr": scheduler.get_last_lr()[0]},
                    step=global_step,
                    prefix="train/"
                )

            # Co 200 batchy – histogram wag
            if batch_idx % 200 == 0:
                for name, param in model.named_parameters():
                    if "weight" in name:
                        logger.log_histogram(f"weights/{name}", param, global_step)

        # Walidacja
        model.eval()
        correct = test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        val_metrics = {"val_loss": test_loss, "val_accuracy": accuracy * 100}
        is_best = accuracy > best_val_acc
        if is_best:
            best_val_acc = accuracy

        logger.log_validation(epoch, val_metrics, is_best=is_best)
        logger.log_epoch_end(epoch, val_metrics, save_checkpoint=True)
        logger.log_metrics(val_metrics, step=global_step, prefix="val/")

        # Checkpoint + LR
        ckpt_path = log_dir / f"checkpoint_epoch_{epoch}.pt"
        logger.log_checkpoint(epoch, global_step, ckpt_path)

        scheduler.step()
        logger.log_learning_rate(scheduler.get_last_lr()[0])

        # Symulacja early stopping w drugim epoku
        if epoch == 2:
            logger.log_early_stopping(epoch, "val_accuracy", accuracy * 100)

    # Końcowe podsumowanie
    logger.training_summary()
    logger.info(f"Najlepsza dokładność: {best_val_acc * 100:.2f}%")
    logger.separate("=", 80)
    logger.header("TEST ZAKOŃCZONY POMYŚLNIE!")

    # Zakończ zewnętrzne loggery
    logger.finish()


if __name__ == "__main__":
    try:
        full_feature_test()
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika")
    except Exception as e:
        import traceback
        traceback.print_exc()