{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/FaiHomjandee/functions/blob/main/Simple_CNN.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install torch torchvision torchaudio --quiet\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torchvision import datasets, transforms"
      ],
      "metadata": {
        "id": "VLQxokXybMsx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 574
        },
        "id": "Kt9EQqtaAbI1",
        "outputId": "c039a459-4277-4bdd-b3f0-67d30c757be7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (2.5.1+cu121)\n",
            "Requirement already satisfied: torchvision in /usr/local/lib/python3.11/dist-packages (0.20.1+cu121)\n",
            "Requirement already satisfied: torchaudio in /usr/local/lib/python3.11/dist-packages (2.5.1+cu121)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch) (3.16.1)\n",
            "Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.11/dist-packages (from torch) (4.12.2)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch) (3.4.2)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.5)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch) (2024.10.0)\n",
            "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch) (12.1.105)\n",
            "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch) (12.1.105)\n",
            "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch) (12.1.105)\n",
            "Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch) (9.1.0.70)\n",
            "Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.11/dist-packages (from torch) (12.1.3.1)\n",
            "Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.11/dist-packages (from torch) (11.0.2.54)\n",
            "Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.11/dist-packages (from torch) (10.3.2.106)\n",
            "Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.11/dist-packages (from torch) (11.4.5.107)\n",
            "Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.11/dist-packages (from torch) (12.1.0.106)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch) (2.21.5)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch) (12.1.105)\n",
            "Requirement already satisfied: triton==3.1.0 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.0)\n",
            "Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch) (1.13.1)\n",
            "Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.11/dist-packages (from nvidia-cusolver-cu12==11.4.5.107->torch) (12.6.85)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch) (1.3.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torchvision) (1.26.4)\n",
            "Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.11/dist-packages (from torchvision) (11.1.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch) (3.0.2)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'\\n# Load MNIST dataset\\ntrain_dataset = datasets.MNIST(\\n    root=\"./data\", train=True, download=True, transform=transforms.ToTensor()\\n)\\ntrain_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)\\n\\n# Create model, optimizer, and loss function\\nmodel = SimpleCNN()\\noptimizer = optim.Adam(model.parameters())\\ncriterion = nn.CrossEntropyLoss()\\n\\n# Training loop\\nnum_epochs = 5\\nfor epoch in range(num_epochs):\\n    for batch_idx, (data, target) in enumerate(train_loader):\\n        optimizer.zero_grad()\\n        output = model(data)\\n        loss = criterion(output, target)\\n        loss.backward()\\n        optimizer.step()\\n\\n        if batch_idx % 100 == 0:\\n            print(\\n                \"Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}\".format(\\n                    epoch + 1,\\n                    batch_idx * len(data),\\n                    len(train_loader.dataset),\\n                    100.0 * batch_idx / len(train_loader),\\n                    loss.item(),\\n                )\\n            )\\n\\nprint(\"Training finished!\")\\n'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 1
        }
      ],
      "source": [
        "class SimpleCNN(nn.Module):\n",
        "    def __init__(self, num_classes=10):\n",
        "        super(SimpleCNN, self).__init__()\n",
        "        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)\n",
        "        self.relu1 = nn.ReLU()\n",
        "        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)\n",
        "        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)\n",
        "        self.relu2 = nn.ReLU()\n",
        "        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)\n",
        "        self.flatten = nn.Flatten()\n",
        "        self.fc1 = nn.Linear(64 * 7 * 7, 128)\n",
        "        self.relu3 = nn.ReLU()\n",
        "        self.fc2 = nn.Linear(128, num_classes)\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.pool1(self.relu1(self.conv1(x)))\n",
        "        x = self.pool2(self.relu2(self.conv2(x)))\n",
        "        x = self.flatten(x)\n",
        "        x = self.relu3(self.fc1(x))\n",
        "        x = self.fc2(x)\n",
        "        return x\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\"\"\"\n",
        "# Load MNIST dataset\n",
        "train_dataset = datasets.MNIST(\n",
        "    root=\"./data\", train=True, download=True, transform=transforms.ToTensor()\n",
        ")\n",
        "train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)\n",
        "\n",
        "# Create model, optimizer, and loss function\n",
        "model = SimpleCNN()\n",
        "optimizer = optim.Adam(model.parameters())\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "\n",
        "# Training loop\n",
        "num_epochs = 5\n",
        "for epoch in range(num_epochs):\n",
        "    for batch_idx, (data, target) in enumerate(train_loader):\n",
        "        optimizer.zero_grad()\n",
        "        output = model(data)\n",
        "        loss = criterion(output, target)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        if batch_idx % 100 == 0:\n",
        "            print(\n",
        "                \"Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}\".format(\n",
        "                    epoch + 1,\n",
        "                    batch_idx * len(data),\n",
        "                    len(train_loader.dataset),\n",
        "                    100.0 * batch_idx / len(train_loader),\n",
        "                    loss.item(),\n",
        "                )\n",
        "            )\n",
        "\n",
        "print(\"Training finished!\")\n",
        "\"\"\""
      ],
      "metadata": {
        "id": "ROWow8btA4fK"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}