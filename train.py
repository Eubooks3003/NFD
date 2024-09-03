import torch
import torch.nn as nn
import torch.optim as optim


import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch
import torch.nn as nn

import torch
import os
from torchvision.utils import save_image


from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(3, 4)
        self.e2 = encoder_block(4, 8)
        self.e3 = encoder_block(8, 16)

        self.b = conv_block(16, 32)

        self.d1 = decoder_block(32, 16)
        self.d2 = decoder_block(16, 8)
        self.d3 = decoder_block(8, 4)

        self.outputs = nn.Conv2d(4, 1, kernel_size=1, padding=0)
    def forward(self, inputs):

        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b = self.b(p3)

        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        outputs = self.outputs(d3)
        return outputs


def frobenius_norm(y_pred, y_true):
    # Reshape tensors to [batch_size, C*H*W]
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)

    norm = torch.norm(y_pred_flat - y_true_flat, p='fro', dim=1)
    squared_frobenius_norm = norm ** 2
    # return squared_frobenius_norm.mean()

    return norm.mean()


class BlockDataset(Dataset):
    def __init__(self, segmented_dir, pose_dir, target_dir, transform=None):
        """
        Args:
            segmented_dir (string): Directory with all the segmented images.
            pose_dir (string): Directory with all pose images (initial and final).
            target_dir (string): Directory with the next state images.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.segmented_dir = segmented_dir
        self.pose_dir = pose_dir
        self.target_dir = target_dir
        self.transform = transform
        self.data = self._prepare_dataset()

    def _prepare_dataset(self):
        files = []
        for filename in os.listdir(self.segmented_dir):
            if filename.endswith('.png'):
                base_name = filename[:-4]  # Remove the .png extension
                initial_pose_path = os.path.join(self.pose_dir, f"{base_name}_Rectange_0.png") # Typo in data processing. Change when new data is collected
                final_pose_path = os.path.join(self.pose_dir, f"{base_name}_Rectange_1.png")
                next_index = int(base_name.split('_')[-1]) + 1  # Assuming the last split part is the index
                next_filename = f"{base_name[:-len(str(next_index - 1))]}{next_index}.png"
                target_img_path = os.path.join(self.target_dir, next_filename)
                
                error = False
                if os.path.exists(initial_pose_path) and os.path.exists(final_pose_path) and os.path.exists(target_img_path):
                    try:
                        #image = Image.open(initial_pose_path).convert('L')
                        print("Trying: ", initial_pose_path)
                    except IOError as e:
                        print("Corrupted Path: ", initial_pose_path)
                        error = True
                    try:
                        #image = Image.open(final_pose_path).convert('L')
                        print("Trying: ", final_pose_path)
                    except IOError as e:
                        print("Corrupted Path: ", final_pose_path)
                        error = True
                    try:
                        #image = Image.open(target_img_path).convert('L')
                        print("Trying: ", target_img_path)
                    except IOError as e:
                        print("Corrupted Path: ", target_img_path)
                        error = True
                    if not error:
                        files.append((filename, initial_pose_path, final_pose_path, target_img_path))
                # else:
                #     print("Initial Pose Path: ", initial_pose_path, " Found: ", os.path.exists(initial_pose_path))
                #     print("Final Pose Path: ", final_pose_path, " Found: ", os.path.exists(final_pose_path))
                #     print("Target Image Path: ", target_img_path, " Found: ", os.path.exists(target_img_path))
        print("Number of Datapoints: ", len(files))
        return files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seg_filename, initial_pose_path, final_pose_path, target_img_path = self.data[idx]

        segmented_image = Image.open(os.path.join(self.segmented_dir, seg_filename)).convert('L')
        initial_pose_image = Image.open(initial_pose_path).convert('L')
        final_pose_image = Image.open(final_pose_path).convert('L')
        target_image = Image.open(target_img_path).convert('L')

        if self.transform:
            segmented_image = self.transform(segmented_image)
            initial_pose_image = self.transform(initial_pose_image)
            final_pose_image = self.transform(final_pose_image)
            target_image = self.transform(target_image)

        # Stack images along the channel axis
        combined_image = torch.cat([segmented_image, initial_pose_image, final_pose_image], dim=0)

        return combined_image, target_image

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),          # Convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 3 stacked grayscale images
])


dataset = BlockDataset(segmented_dir='/home/ellina/Working/NFD/output_masks_red',
                       pose_dir='/home/ellina/Working/NFD/output_actions_test',
                       target_dir='/home/ellina/Working/NFD/output_masks_red',
                       transform=transform)


# Assuming 'dataset' is already loaded and ready to be split
train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
val_size = len(dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

from tqdm import tqdm
import torch
from torchvision.utils import save_image
import os

import os
import shutil

def setup_and_clean_directory(directory):
    """Checks if the directory exists and creates it if not, then clears any existing files."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def initialize_folders(base_name):
    """Creates and cleans the directories for training and validation data."""
    base_dirs = {
        'training': ['output_images_', 'input_images_', 'target_images_'],
        'validation': ['output_images_', 'input_images_', 'target_images_']
    }
    paths = {}
    for key, dirs in base_dirs.items():
        for dir in dirs:
            full_path = os.path.join(base_name, key, dir)
            setup_and_clean_directory(full_path)
            paths[dir + key] = full_path  # Store path with key for easy access, e.g., "output_images_training"
    return paths


def save_epoch_images(epoch, i, outputs, inputs, targets, paths, phase='training'):
    """Saves output, input, and target images using the paths dictionary based on phase."""
    save_image(outputs, os.path.join(paths[f'output_images_{phase}'], f'outputs_epoch_{epoch+1}_{i}.png'))
    save_image(inputs, os.path.join(paths[f'input_images_{phase}'], f'inputs_epoch_{epoch+1}_{i}.png'))
    save_image(targets, os.path.join(paths[f'target_images_{phase}'], f'targets_epoch_{epoch+1}_{i}.png'))

def validate_and_save(model, dataloader, epoch, loss_function, device, paths):
    model.eval()
    total_loss = 0
    val_progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1} Validation', leave=False)
    with torch.no_grad():
        for i, (inputs, targets) in val_progress:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            

            if i % 20 == 0:  # Save first batch output for visual inspection
            
                save_epoch_images(epoch, i, outputs, inputs, targets, paths, phase='validation')

    average_loss = total_loss / len(dataloader)
    print(f'\nValidation Loss: {average_loss:.4f}')
    return average_loss

# Model, Loss, and Optimizer
model = Unet()  # Assuming you are stacking three images
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 20
best_loss = float('inf')
validate_every_n_epochs = 2

criterion = nn.MSELoss()

base_name = "train_data"
paths = initialize_folders(base_name)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs, validate_every_n_epochs, base_name):
    writer = SummaryWriter(log_dir=f'{base_name}/logs')  # TensorBoard writer
    best_loss = float('inf')
    paths = initialize_folders(base_name)

    for epoch in range(num_epochs):
        model.train()
        train_progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch+1} Training')
        for i, (inputs, targets) in train_progress:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                save_epoch_images(epoch, i, outputs, inputs, targets, paths)

            # Log training loss to TensorBoard
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_dataloader) + i)

            train_progress.set_postfix(loss=loss.item())

        scheduler.step()

        if (epoch + 1) % validate_every_n_epochs == 0:
            val_loss = validate_and_save(model, val_dataloader, epoch, criterion, device, paths)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'{base_name}/model_checkpoint_best.pth')
                print(f'Saved best model checkpoint at epoch {epoch+1}.')
                # Log validation loss to TensorBoard
                writer.add_scalar('Validation Loss', val_loss, epoch)

    writer.close()  # Close the TensorBoard writer

def hyperparameter_sweep(train_dataloader, val_dataloader, device, epochs, validate_every):
    learning_rates = [0.01, 0.001, 0.0001]
    step_sizes = [5, 10]
    gammas = [0.1, 0.2]
    criteria = [nn.MSELoss(), frobenius_norm] 

    for lr in learning_rates:
        for step_size in step_sizes:
            for gamma in gammas:
                for criterion in criteria:
                    criterion_name = 'MSE' if isinstance(criterion, nn.MSELoss) else 'Frobenius'
                    base_name = f"lr{lr}_step{step_size}_gamma{gamma}_{criterion_name}"
                    print(f"Training with LR: {lr}, Step Size: {step_size}, Gamma: {gamma}, Criterion: {criterion_name}")
                    model = Unet().to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
                    criterion = nn.MSELoss()

                    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs, validate_every, base_name)

# hyperparameter_sweep(train_dataloader, val_dataloader, device, num_epochs, validate_every_n_epochs)
lr = 0.01
gamma = 0.1
step_size = 5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
train_model(model, train_dataloader, val_dataloader, nn.MSELoss(), optimizer, scheduler, device, num_epochs, validate_every_n_epochs, "micro-actions-8")