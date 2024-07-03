import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        """
        The double convolution layer used per block in UNet Architecture
        Args:
            in_channels: # of channels in input image. E.g 3 channels for RGB images.
            out_channels: # of filters applied to input image. Each filter detects different features from input
        """
        super(DoubleConv, self).__init__()
        # Sequential module to map out the double conv layer per block in UNET Architecture
        self.double_conv = nn.Sequential(
            # Kernel size of 3, stride 1, and padding 1 will help achieve same convolution
            # Bias is set to false since BatchNorm will just cancel it out
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # ReLU activation will apply non-linearity after convolution and maintain it during normalization
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 5, features = [64, 128, 256, 512]):
        """
        Args:
            in_channels: Begin with 3 for an RGB image.
            out_channels: Begin with 5 for the categories: shirts, pants, dresses, accessories, shoes
            features: Feature mapping for downsampling and upsampling in the paths of the UNet architecture
        """
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Pooling layer used for down sampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path (encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck layer
            # Connecting the final 512 layer in downsampling path to the bottleneck 1024 layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Upsampling path (decoder)
        for feature in reversed(features):
            # Use bilinear upsampling and a conv layer
            self.ups.append(
                nn.Sequential(
                    # Double the size of input feature map with bilinear calculations
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.Dropout(p=0.2)
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Final 1x1 conv layer
            # Adjusts the depth of the output feature maps to match # of desired classes
            # Essentially matching the initial out_channels to this final layer 
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward step/pass is the process by which input data is passed through the network
        from the first to the last layer to produce an output.
        
        Args:
            x: Input tensor to the network (image that is passed into the model for processing)
        """
        skips = []

        # Downsampling path
            # Begin at the highest resolution channel down to the smallest
        for down in self.downs:
            x = down(x) # Applies the double conv to the input tensor
            skips.append(x) # Saves the output for the skip connection (grey arrows)
            x = self.pool(x) # Apply the max pooling (downsample) to the feature map 

        # Applies further conv without any downsampling
        x = self.bottleneck(x)

        skips = skips[::-1] # Reverse the skip connections to align in upsampling path

        # Upsampling path
            # Skipping by 2 each time as we are doing the up conv and double conv in 1 step
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # Upsample (the sequential bilinear sampling) the input tensor's feature map
            skip = skips[i // 2] # Retrieve corresponding feature map from reversed skip connections
            
            # If the upsampled feature map and the skip feature map don't match in shape,
            # adjust the size of the upsampled feature map to match the skip feature map
                # This could happen when we downsample with max pool. E.g. 161 x 161 -> 80 x 80
                # then when we upsample in the expanding path, it may go up to 160 x 160.
                # If this happens, we wont be able to concat as they need same height and width
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])
            
            x = torch.cat((skip, x), axis=1) # Concatenate the skip map with the upsampled map along channel dimension
            x = self.ups[i + 1](x) # Upsample normally to integrate concatenated feature maps

        # Apply a final convolution to map the features to the number of desired output channels/classes
            # No need to apply Softmax here since we are using cross entropy loss when training which already uses softmax activation internally
        return self.final_conv(x)

def testMultiClassRGB():
    x = torch.randn((3, 3, 161, 161)) # Random input tensors of a batch of three 161x161 RGB images 

    model = UNet(in_channels=3, out_channels=5) # 5 categories/classes
    predictions = model(x) # Perform convolutions

    # Check if batch size and spatial dimensions are correct
    assert predictions.shape[0] == x.shape[0]
    assert predictions.shape[2:] == x.shape[2:]
    assert predictions.shape[1] == 5  # Ensure the output channels match the number of classes


def testBinarySegmentationGrayscale():
    x = torch.randn((3, 1, 161, 161))

    model = UNet(in_channels=1, out_channels=1)
    predictions = model(x)
    assert predictions.shape == x.shape

if __name__ == "__main__":
    testMultiClassRGB()
    testBinarySegmentationGrayscale()