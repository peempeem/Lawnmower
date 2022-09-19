import torch
import numpy as np
import cv2
from tqdm import tqdm
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

from dataset import ImageDim, DataLoader, TrainingDataset, ImageDataset
from model import GNetModel


class GNet:
    def __init__(self,
            img_dims,
            class_path, color_path,
            load_model=None,
            device="cuda",
            batch_size=1,
            workers=1,
            pin_memory=False):

        self._img_dims = img_dims

        self._classes = load_classes(class_path)
        self._colors = load_colors(color_path)

        assert self._classes != []
        assert self._colors != []
        assert len(self._classes) == len(self._colors)

        self._device = device
        self.batch_size = batch_size
        self.workers = workers
        self.pin_memory = pin_memory
        self._model = GNetModel(in_channels=self._img_dims.get_channels(), out_channels=len(self._classes)).to(self._device)

        if load_model is not None:
            print(f"Loading model from: {load_model}")
            load_checkpoint(load_model, self._model)

        self._model.eval()
        temp = np.zeros(self._img_dims.get_BCHW(), dtype=np.float32)
        temp = torch.tensor(temp).to(self._device)
        preds = self._model(temp)

        _, o_c, o_h, o_w = preds.shape
        self._out_dims = ImageDim(o_w, o_h, o_c)

        print("\n[MODEL PARAMETERS]")
        print(f"Classes: {self._classes}")
        print(f"Input Dimensions: {img_dims.get_BCHW()}")
        print(f"Output Dimensions: {self._out_dims.get_BCHW()}")
        print(f"Device: {self._device}\n")

    def summary(self):
        summary(self._model, self._img_dims.get_CHW())

    def train(self,
              trainDB_path,
              valDB_path,
              epochs=1,
              learning_rate=1e-4,
              saveas="model.pth.tar",
              shuffle=False):

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler()

        train_ds = TrainingDataset(
            f"{trainDB_path}/images",
            f"{trainDB_path}/masks",
            self._img_dims,
            self._out_dims,
            self._colors
        )

        train_loader = DataLoader(train_ds,
            batch_size=self.batch_size,
            workers=self.workers,
            shuffle=shuffle)

        val_ds = TrainingDataset(
            f"{valDB_path}/images",
            f"{valDB_path}/masks",
            self._img_dims,
            self._out_dims,
            self._colors
        )

        val_loader = DataLoader(val_ds,
            batch_size=self.batch_size,
            workers=self.workers
        )

        print("\n[TRAINING PARAMETERS]")
        print(f"Epochs: {epochs}")
        print(f"Image Count: {len(train_ds)}")
        print("Optimizer: ADAM")
        print(f"Learning Rate: {learning_rate}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Workers: {self.workers}")

        print(f"\nStarting training for {epochs} epochs.")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self._model.train()

            loop = tqdm(train_loader, desc="Progress")

            for batch_idx, (data, targets) in enumerate(loop):
                data = data.to(device=self._device)
                targets = targets.long().to(device=self._device)

                with torch.cuda.amp.autocast():
                    predictions = self._model(data)
                    loss = loss_fn(predictions, targets)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loop.set_postfix(loss=loss.item())

            print("")
            self.check_accuracy(val_loader)
            print("\n")

        if saveas is not None:
            print(f"Saving model as \"{saveas}\"")

            checkpoint = {
                "state_dict": self._model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=saveas)

    def toONNX(self, filename):
        x = torch.ones(self._img_dims.get_BCHW()).to(self._device)
        torch.onnx.export(self._model, x, filename, input_names=['input'],
            output_names=['output'], export_params=True)

    def predict_folder(self, image_folder):
        ds = ImageDataset(image_folder, self._img_dims)
        loader = DataLoader(ds, batch_size=self.batch_size, workers=self.workers)

        print(f"\nPredicting {len(ds)} images...")
        loop = tqdm(loader, desc="Progress")

        images = []
        masks = []

        self._model.eval()
        for batch_idx, tensors in enumerate(loop):
            device_tensors = tensors.to(device=self._device)
            with torch.no_grad():
                preds = self._model(device_tensors).cpu().numpy()
                preds = np.transpose(preds, (0, 2, 3, 1))
                preds = np.argmax(preds, axis=3)

                for i in range(len(device_tensors)):
                    height, width = preds[i].shape
                    arr = np.zeros((height, width, self._img_dims.get_channels()))
                    for y in range(height):
                        for x in range(width):
                            arr[y, x] = self._colors[preds[i, y, x]]
                    mask = cv2.cvtColor(arr.astype(np.float32) / 255, cv2.COLOR_RGB2BGR)
                    mask = cv2.resize(mask, self._img_dims.get_WH(), interpolation=cv2.INTER_NEAREST)

                    image = np.transpose(tensors[i].numpy().astype(np.float32), (1, 2, 0))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    images.append(image)
                    masks.append(mask)
        return images, masks

    def show_images(self, images, masks, scale=2, delay=500):
        for i in range(len(images)):
            out_image = np.concatenate((images[i], masks[i]), axis=1)
            height, width = out_image.shape[:2]
            out_image = cv2.resize(out_image, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('', out_image)
            cv2.waitKey(delay)

    def _train_fn(self, loader, optimizer, loss_fn, scaler):
        loop = tqdm(loader, desc="Progress")

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self._device)
            targets = targets.long().to(device=self._device)

            with torch.cuda.amp.autocast():
                predictions = self._model(data)
                loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

    def check_accuracy(self, loader):
        print("Checking Accuracy...")
        num_correct = 0
        num_pixels = 0
        loop = tqdm(loader, desc="Progress")

        self._model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loop):
                x = x.to(self._device)
                y = y.cpu()
                preds = self._model(x).cpu()
                preds = np.argmax(preds, axis=1)
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)

        print(f"Pixels Correct: {num_correct}/{num_pixels}")
        print(f"Accuracy: {num_correct/num_pixels*100:.2f}%")


def load_classes(path):
    classes = []
    file = open(path, 'r')
    for line in file:
        if '\n' in line:
            line = line.replace('\n', '')
        classes.append(line)
    return classes


def load_colors(path):
    colors = []
    file = open(path, 'r')
    for line in file:
        if '\n' in line:
            line = line.replace('\n', '')
        if ' ' in line:
            line = line.replace(' ', '')
        numbers = line.split(',')
        for i in range(0, len(numbers)):
            numbers[i] = int(numbers[i])
        colors.append(tuple(numbers))
    return colors


def load_checkpoint(checkpoint, model):
    checkpt = torch.load(checkpoint)
    model.load_state_dict(checkpt["state_dict"])


def save_checkpoint(state, filename="model.pth.tar"):
    torch.save(state, filename)
