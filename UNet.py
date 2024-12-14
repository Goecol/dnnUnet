import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torchvision.transforms import CenterCrop
import torchvision.transforms.functional as Ff

# The custom UNet module is defined as a subclass of nn.Module. 
# The __init__ method (which is the constructor of the defined UNet module) initializes the architecture of the U-Net by defining the layers for both the encoder and decoder parts of the network. 
# The argument num_of_classes specifies the number of classes for the segmentation task. 
# When this UNet module is called, the argument (num_of_classes) must be passed
class UNet(nn.Module):
    def __init__(self, num_of_classes, num_of_out_channels):
        super().__init__()
        
        # ------------------ ENCODER ---------------------------------------------------
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.en11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.en12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.en21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.en22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.en31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.en32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.en41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.en42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.en51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        #self.en52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024
        self.en52 = self.en51


        # ------------------ DECODER ---------------------------------------------------
        # Decoder
        # In the decoder, transpose convolutional layers with the ConvTranspose2d function are used to upsample the feature maps to the original size of the input image. 
        # Each block in the decoder consists of an upsampling layer, a concatenation with the corresponding encoder feature map, and two convolutional layers.
        # -------
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2)  # output: 56x56x512
        self.dec11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)  # output: 54x54x512
        self.dec12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)   # output: 52x52x512

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2) # output: 104x104x256
        self.dec21 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # output: 102x102x256
        self.dec22 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 100x100x256

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # output: 200x200x128
        self.dec31 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # output: 198x198x128
        self.dec32 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 196x196x128

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # output: 392x392x64
        self.dec41 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # output: 390x390x64
        self.dec42 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 388x388x64

        # Output layer
        #self.outconv = nn.Conv2d(64, num_of_classes, kernel_size=1) # output: 388x388x(num_of_classes)
        self.outconv = nn.Conv2d(64, num_of_out_channels, kernel_size=1) # output: 388x388x(num_of_classes)

   
    def forward(self, x):
        # Encoder
        xen11 = relu(self.en11(x)) # x output for the encoder11
        xen12 = relu(self.en12(xen11))
        xp1 = self.pool1(xen12) # x output for the pooling layer

        xen21 = relu(self.en21(xp1))
        xen22 = relu(self.en22(xen21))
        xp2 = self.pool2(xen22)

        xen31 = relu(self.en31(xp2))
        xen32 = relu(self.en32(xen31))
        xp3 = self.pool3(xen32)

        xen41 = relu(self.en41(xp3))
        xen42 = relu(self.en42(xen41))
        xp4 = self.pool4(xen42)

        xen51 = relu(self.en51(xp4))
        #xen52 = relu(self.en52(xen51))
        xen52 = xen51
        
        # Decoder
        xu1 = self.upconv1(xen52)   # expected output: 56x56x1024
        cropped_xen42 = test.check_and_crop_tensor2(xu1, xen42)
        xu11 = torch.cat([xu1,  xen42], dim=1)  # old code 
        xdec11 = relu(self.dec11(xu11))
        xdec12 = relu(self.dec12(xdec11))

        xu2 = self.upconv2(xdec12)
        cropped_xen32 = test.check_and_crop_tensor2(xu2, xen32)
        xu22 = torch.cat([xu2, xen32], dim=1)
        xdec21 = relu(self.dec21(xu22))
        xdec22 = relu(self.dec22(xdec21))

        xu3 = self.upconv3(xdec22)
        cropped_xen22 = test.check_and_crop_tensor2(xu3, xen22)
        xu33 = torch.cat([xu3, xen22], dim=1)
        xdec31 = relu(self.dec31(xu33))
        xdec32 = relu(self.dec32(xdec31))

        xu4 = self.upconv4(xdec32)
        cropped_xen12 = test.check_and_crop_tensor2(xu4, xen12)
        xu44 = torch.cat([xu4, xen12], dim=1)
        xdec41 = relu(self.dec41(xu44))
        xdec42 = relu(self.dec42(xdec41))

        # Output layer
        out = self.outconv(xdec42)

        return out



import test

test.run(UNet)
