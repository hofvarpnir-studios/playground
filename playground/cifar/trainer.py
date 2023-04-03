from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from playground.cifar.utils import acc
from playground.cifar.models import ConvNet, MLPNet
from playground.tracker.tracker import Tracker
import gin


@gin.configurable
class CIFARData:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.categories = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )
        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )
        self.testloader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    @staticmethod
    def imshow(img: torch.Tensor):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def sample(self):
        iterator = iter(self.trainloader)
        images, labels = iterator.next()
        self.imshow(torchvision.utils.make_grid(images))
        print(
            " ".join("%5s" % self.categories[labels[j]] for j in range(self.batch_size))
        )


class Trainer:
    def __init__(
        self,
        epochs: int,
        data: CIFARData,
        network: Union[MLPNet, ConvNet],
        lr: float,
        tracker: Tracker,
    ):
        self.epochs = epochs
        self.data_class = data
        self.network = network
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.network = torch.compile(self.network)
        self.tracker = tracker

    def train(self):
        print("Training")
        train_results = []
        validation_results = []
        self.network.train()
        self.tracker.set_conf()
        k = 0
        for _ in range(self.epochs):
            running_loss = 0.0
            correct, total = 0, 0
            for i, (images, labels) in enumerate(tqdm(self.data_class.trainloader)):
                batch_size = images.size(0)
                # get network output
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.network(images)

                # compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += batch_size
                correct += (predicted == labels).sum().item()

                # do gradient descent
                self.optimizer.zero_grad()
                loss: torch.Tensor = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # add loss
                running_loss += loss.item()
                self.tracker.log_iteration_time(batch_size, k)
                if i % 100 == 99:  # print every 100 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f acc: %.3f"
                        % (k, i + 1, running_loss / 100, acc(correct, total))
                    )
                    self.tracker.add_scalar("metrics/train-loss", running_loss / 100, k)
                    running_loss = 0.0
                    train_results.append((k, acc(correct, total)))
                    self.tracker.add_scalar("metrics/train-acc", acc(correct, total), k)
                    correct, total = 0, 0
                    test_loss, test_accuracy = self.validate_accuracy()
                    validation_results.append((k, test_loss / 100, test_accuracy))
                    self.tracker.add_scalar("metrics/test-loss", test_loss / 100, k)
                    self.tracker.add_scalar("metrics/test-acc", test_accuracy, k)
                k += 1

    def validate_accuracy(self) -> Tuple[float, float]:
        print("Validating accuracy")
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(self.data_class.testloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.network(images)
                loss: torch.Tensor = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss, acc(correct, total)

    def validate_by_class(self) -> Dict[str, float]:
        print("Validating by class")
        self.network.eval()
        correct_pred = {classname: 0 for classname in self.data_class.categories}
        total_pred = {classname: 0 for classname in self.data_class.categories}

        with torch.no_grad():
            for images, labels in tqdm(self.data_class.testloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.network(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.data_class.categories[label]] += 1
                    total_pred[self.data_class.categories[label]] += 1
        class_accuracy: Dict[str, float] = {}
        for classname, correct_count in correct_pred.items():
            accuracy = acc(correct_count, total_pred[classname])
            class_accuracy[classname] = accuracy
        self.tracker.add_dictionary({"class_acc": class_accuracy})
        return class_accuracy
