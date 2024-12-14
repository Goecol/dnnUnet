import cv2
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import torchvision.transforms.functional as Ff
import os
import time 
import torch.nn as nn


train_dir = "/home/johnbosco/soybean/dataset_UNET/train/001.lessthan40/"
train_mask_dir = "/home/johnbosco/soybean/dataset_UNET/pre_processed/train/001.lessthan40/"
val_dir = "/home/johnbosco/soybean/dataset_UNET/val/001.lessthan40/"
val_mask_dir = "/home/johnbosco/soybean/dataset_UNET/pre_processed/val/001.lessthan40/"
test_dir = "/home/johnbosco/soybean/dataset_UNET/test/001.lessthan40/"
test_mask_dir = "/home/johnbosco/soybean/dataset_UNET/pre_processed/test/001.lessthan40/"
batch_size = 1
train_loss = []
val_loss = []

'''
train_dir = "/home/johnbosco/soybean/dataset_UNET/train/002.41to80/"
train_mask_dir = "/home/johnbosco/soybean/dataset_UNET/pre_processed/train/002.41to80/"
val_dir = "/home/johnbosco/soybean/dataset_UNET/val/002.41to80/"
val_mask_dir = "/home/johnbosco/soybean/dataset_UNET/pre_processed/val/002.41to80/"
test_dir = "/home/johnbosco/soybean/dataset_UNET/test/002.41to80/"
test_mask_dir = "/home/johnbosco/soybean/dataset_UNET/pre_processed/test/002.41to80/"
'''


torch.cuda.empty_cache()


class SoybeanPodDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Args:
            image_paths (list of str): List of file paths to input images
            mask_paths (list of str): List of file paths to corresponding masks
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        self.image_paths = sorted(self.image_paths)  # Sort image file paths
        self.mask_paths = sorted(self.mask_paths)    # Sort label mask file paths

        # Load the image and mask
        #image = Image.open(self.image_paths[idx]).convert("RGB")
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx]).convert("L")  # 1-channel (grayscale)

        mask_trans = transforms.Compose([
        transforms.Resize((572, 572)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # imagenet   #one channel
          ])


        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)
            mask = mask_trans(mask)

        return image, mask



def convert_numpy_array(array):
    array = array.squeeze()  # Now the shape is (3,)

    # Check if the values are in the range [0, 1], multiply by 255 if necessary
    if array.max() <= 1:
        array = (array * 255).astype(np.uint8)  # S

    return array


def convert_mask_to_rgb(mask, foreground_color=(255, 255, 0), background_color=(0, 0, 0)):
    """
    Convert a binary mask (0s and 1s) to an RGB mask for visualization.
    Args:
    - mask (Tensor): The binary mask of shape (height, width)
    - foreground_color (tuple): RGB values for foreground (soybean pods)
    - background_color (tuple): RGB values for background

    Returns:
    - rgb_mask (Tensor): The RGB mask of shape (height, width, 3)
    """
    # Create an RGB mask with the same height and width but 3 channels (RGB)
    rgb_mask = torch.zeros(mask.shape[0], mask.shape[1], 3, dtype=torch.uint8)
    
    # Set the background pixels
    rgb_mask[mask == 0] = torch.tensor(background_color, dtype=torch.uint8)
    
    # Set the foreground (soybean pods) pixels
    rgb_mask[mask == 1] = torch.tensor(foreground_color, dtype=torch.uint8)
    
    return rgb_mask


def tensor_to_image(tensor):
    """Converts a PyTorch tensor to a PIL Image."""
    # Ensure the tensor is in the correct range [0, 1]
    tensor = tensor.clamp(0, 1)

    # Convert the tensor to a numpy array
    tensor = tensor.cpu().detach().numpy()

    # Transpose the dimensions to (height, width, channels)
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    tensor = tensor.transpose((1, 2, 0))

    # Convert the numpy array to a PIL Image
    image = Image.fromarray((tensor * 255).astype('uint8'))
    return image


def get_image_filepaths(directory):
    """Returns a list of filepaths for all images in the given directory."""

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    image_paths = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and any(filename.endswith(ext) for ext in image_extensions):
            image_paths.append(filepath)

    return image_paths

def convert_tensor(label_tensor, predicted_tensor):
    top, left, height, width = get_bounding_box_values(predicted_tensor)

    # Desired shape for the predicted tensor
    batch_size = top
    num_classes = left
    height = height
    width = width

    labels = label_tensor
    print(label_tensor.shape)
    print(predicted_tensor.shape)
    # Convert the label tensor to match the predicted tensor's size [5, 2, 388, 388]
    expanded_labels = labels.view(batch_size, 1, 1, 1).expand(-1, num_classes, height, width)

    expanded_labels = expanded_labels.float()

    # Check the shape of the expanded labels tensor
    print(expanded_labels.shape)  # Should be [5, 2, 388, 388]
    return expanded_labels


def convert_tensor2(label_tensor, predicted_tensor):
    top, left, height, width = get_bounding_box_values(predicted_tensor)

    # Desired shape for the predicted tensor
    batch_size = top
    num_classes = left   # number classes or number of channels
    height = height
    width = width


    # Check if the input tensor has the expected shape
    # assert label_tensor.shape == (5, 3, 572, 572), "Input tensor shape is incorrect"
    # assert label_tensor.shape == (batch_size, num_classes, height, width), "Input tensor shape is incorrect"

    # Change the number of channels from 3 to 2
    # We'll assume you want to keep the first 2 channels
    new_label_tensor = label_tensor[:, :num_classes, :, :] 

    # Resize the tensor using interpolation
    new_label_tensor = F.interpolate(new_label_tensor, size=(height, width), mode='bilinear', align_corners=False)

    return new_label_tensor



def resize_label_tensor(label_tensor, predicted_tensor):
    """Resizes the label tensor to match the size of the predicted tensor."""
    
    #resized_label_tensor = F.interpolate(label_tensor, size=predicted_tensor.shape[2:], mode='nearest')
    resized_label_tensor = label_tensor.repeat(predicted_tensor.shape[0], predicted_tensor.shape[1], predicted_tensor.shape[2], 1)
    resized_label_tensor = F.interpolate(resized_label_tensor, size=predicted_tensor.shape[2:], mode='nearest')

    return resized_label_tensor


def crop_tensor(tensor, top, left, height, width):
        """Crops a tensor to the specified region.

        Args:
            tensor (torch.Tensor): The input tensor.
            top (int): The top coordinate of the crop box.
            left (int): The left coordinate of the crop box.
            height (int): The height of the crop box.
            width (int): The width of the crop box.

        Returns:
            torch.Tensor: The cropped tensor.
        """
        return Ff.crop(tensor, top, left, height, width)

def get_bounding_box_values(tensor):
        """
        Gets the top, left, length, and width values of a tensor shape.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing (top, left, length, width).
        """
        top, left, height, width = tensor.shape[0:]

        top = top
        left = left
        length = height
        width = width
        return top, left, length, width


def check_and_crop_tensor(tensor_target, original_tensor):
        top, left, height, width = get_bounding_box_values(tensor_target)

        top = top
        left = left
        height = height
        width = width

        cropped_tensor = crop_tensor(original_tensor, top, left, height, width)
        return cropped_tensor

def check_and_crop_tensor1(tensor_target, original_tensor):
        xu1 = tensor_target
        xen42 = original_tensor
        if xu1.size(2) != xen42.size(2) or xu1.size(3) != xen42.size(3):
                delta_height = xu1.size(2) - xen42.size(2)
                delta_width = xu1.size(3) - xen42.size(3)
                xen42 = F.pad(xen42, (0, delta_width, 0, delta_height))

        return xen42 

def check_and_crop_tensor2(tensor_target, original_tensor):
        xu1 = tensor_target
        xen42 = original_tensor
        if xu1.size(2) != xen42.size(2) or xu1.size(3) != xen42.size(3):
            xen42 = xen42[:, :, :xu1.size(2), :xu1.size(3)]

        return xen42 

def check_and_crop_tensor4(tensor_target, original_tensor):
        top, left, height, width = get_bounding_box_values(tensor_target)

        top = top
        left = left
        height = height
        width = width

        cropped_tensor = convert_tensor2(original_tensor, tensor_target)
        return cropped_tensor

def getDevice():
    device = None
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_on_gpu = torch.cuda.is_available()
    #train_on_gpu = torch.backends.mps.is_available()
    #train_on_gpu = False

    if not train_on_gpu:
        print('CUDA/MPS is not available.  Training on CPU ...')
        device = torch.device("cpu")
    else:
        print('CUDA/MPS is available!  Training on GPU ...')
        #device = torch.device("mps")
        device = torch.device("cuda")
    
    return device

   
def numpy_to_image(array, image_file_name):
        image = Image.fromarray(array)
        # Save the image
        image.save(image_file_name)
        # Display the image
        #image.show()

def get_data_loaders():
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.Resize((572, 572)),
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # imagenet
    ])

    mask_trans = transforms.Compose([
        transforms.Resize((572, 572)),
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485], [0.229]) # imagenet   #one channel
        transforms.Normalize([0.5], [0.5]) 
    ])
    
    # Create dataset and dataloaders
    # Train loader
    # train_dir = "/path/to/my/images"
    image_paths = get_image_filepaths(train_dir)
    mask_paths = get_image_filepaths(train_mask_dir)

    print(image_paths)
    print(mask_paths)
    train_dataset = SoybeanPodDataset(image_paths, mask_paths, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #shuffle=True

    # Create dataset and dataloaders
    # Validation loader
    val_image_paths = get_image_filepaths(val_dir)
    val_mask_paths = get_image_filepaths(val_mask_dir)
    val_dataset = SoybeanPodDataset(val_image_paths, val_mask_paths, transform=trans)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) #shuffle=True


    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    return dataloaders


def get_dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    print("PRED:")
    print(pred.shape)
    print("TARGET:")
    print(target.shape)
    #target = convert_tensor2(target, pred)
    print(target.shape)
    
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)   # F.sigmoid(pred)
    dice = get_dice_loss(pred, target)

    print(bce)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] +=  bce.data.cpu().detach().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().detach().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().detach().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))



def train_model(model, optimizer, scheduler, num_epochs=25):
    dataloaders = get_data_loaders()
    device = device = getDevice()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    '''
    criterion = torch.nn.BCEWithLogitsLoss().to(device)  # For binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    '''
    running_loss = 0.0
    running_loss_2 = 0.0
    
    start = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        epoch_loss_train = 0.0
        epoch_loss_val = 0.0

        # Each epoch has a training and validation phase
        #for phase in ['train', 'val']:
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #labels = test.check_and_crop_tensor(inputs, labels)
                print(inputs.shape)
                print(labels.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                print("Training............")
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)     # old value
                    loss = calc_loss(outputs, labels, metrics)  # old value

                    '''
                    outputs = model(inputs)
                    target_ = convert_tensor2(labels, outputs)
                    loss = criterion(outputs, target_)
                    '''

                    print("loss:")
                    print(loss)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        #loss.data.cpu().detach().numpy() * target.size(0)
                        epoch_loss_train += loss.item()
                        #running_loss_2 += loss.data.cpu().numpy() * labels.size(0)
                    else:
                        epoch_loss_val += loss.item()
                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
              avg_train_loss = epoch_loss_train/len(dataloaders['train'])
              train_loss.append(avg_train_loss)
              print(f"Train Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
            else:
              avg_val_loss = epoch_loss_val/len(dataloaders['val']) 
              val_loss.append(avg_val_loss)
              print(f"Val Epoch [{epoch+1}/{num_epochs}], Loss: {avg_val_loss:.4f}")

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    end = time.time()
    time_to_train = (end - start)
    print("Time to train: "+ str(time_to_train) + " seconds")

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "my_trained_model.pth")
    return model


def run(UNet):
    in_channels = 3
    num_class = 2
    num_out_channels = 1 # for binary segmentation, the output channel = 1
    batch_size = 1
    num_epochs = 2
    device = getDevice()

    model = UNet(num_class, num_out_channels).to(device)
    #model = UNet2(in_channels, num_out_channels).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

    # Plot train and validation loss
    plt.plot(range(1, num_epochs + 1), train_loss, label="Train Loss", color='blue')
    plt.plot(range(1, num_epochs + 1), val_loss, label="Validation Loss", color='red')
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Validation Loss')
    plt.legend()
    # Show the plot
    plt.show()

    # Perform model testing and prediction
    model.eval()  # Set model to the evaluation mode

    trans = transforms.Compose([
        transforms.Resize((572, 572)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # imagenet
    ])
    # # Create another simulation dataset for test
    test_image_paths = get_image_filepaths(test_dir)
    test_mask_paths = get_image_filepaths(test_mask_dir)

    test_dataset = SoybeanPodDataset(test_image_paths, test_mask_paths, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    i = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:  # Replace test_loader with your test data
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Apply sigmoid to get probabilities and threshold to get binary mask
            predicted_mask = torch.sigmoid(outputs) 
            
            # Visualize the result
            plt.imshow(predicted_mask[0, 0].cpu().numpy(), cmap='gray')
            plt.show()

            numpy_input = inputs[i].permute(1,2,0).cpu().numpy()
            numpy_label = labels[i].cpu().squeeze().numpy()
            numpy_predicted = predicted_mask[0,0].cpu().numpy()

            # Plot the images in a single row with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Show the test image     
            axes[0].imshow(numpy_input)
            axes[0].set_title("Test Image")
            #axes[0].axis('off')  # Hide axis for better visualization

            # Show the label mask
            axes[1].imshow(numpy_label, cmap='gray')
            axes[1].set_title("Label Mask")
            #axes[1].axis('off')  # Hide axis

            # Show the predicted mask
            axes[2].imshow(numpy_predicted, cmap='gray')
            axes[2].set_title("Predicted Mask")
            #axes[2].axis('off')  # Hide axis

            # Display the plot
            plt.tight_layout()
            plt.show()

        

