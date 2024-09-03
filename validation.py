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

# Assuming the model architecture is already defined
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

# Function to calculate MSE loss over a dataset
def calculate_mse_loss(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in tqdm(dataloader, desc="Calculating Loss"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)  # Accumulate loss, scaled by batch size
    
    average_loss = total_loss / len(dataloader.dataset)  # Calculate average loss
    return average_loss

# Load the checkpointed model
def load_model_and_calculate_loss(checkpoint_path, train_loader, val_loader, device):
    # Initialize the model and load the state_dict
    model = Unet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Define the criterion
    criterion = nn.MSELoss()

    # Calculate MSE loss on the training set
    train_loss = calculate_mse_loss(model, train_loader, criterion, device)
    print(f'Training MSE Loss: {train_loss:.4f}')

    # Calculate MSE loss on the validation set
    val_loss = calculate_mse_loss(model, val_loader, criterion, device)
    print(f'Validation MSE Loss: {val_loss:.4f}')

    return train_loss, val_loss


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

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '/home/ellina/Working/NFD/micro-actions/model_checkpoint_best.pth'

# Assuming train_loader and val_loader are already defined DataLoader instances
train_loss, val_loss = load_model_and_calculate_loss(checkpoint_path, train_dataloader, val_dataloader, device)
