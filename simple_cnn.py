args['img_size'] = 256
args['num_classes'] = 2

class SimpleCNN(nn.Module):
    def __init__(self):

        self.img_size = args['img_size']
        self.num_classes = args['num_classes']

        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the output size after convolutional and pooling layers
        # Assuming input size is 256x256
        self.output_size = self._calculate_output_size(img_size,img_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def _calculate_output_size(self, height, width):
        """Calculates the output size after convolutional and pooling layers."""
        # Apply conv1 and pool1
        height = (height + 2 * 1 - 3) // 1 + 1  # (input_size + 2*padding - kernel_size) // stride + 1
        width = (width + 2 * 1 - 3) // 1 + 1
        height //= 2  # MaxPool2d with stride 2
        width //= 2
        
        # Apply conv2 and pool2
        height = (height + 2 * 1 - 3) // 1 + 1
        width = (width + 2 * 1 - 3) // 1 + 1
        height //= 2
        width //= 2
        
        # Return the flattened size
        return height * width * 64 # 64 is the number of output channels from conv2
