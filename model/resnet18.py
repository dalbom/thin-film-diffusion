import torch.nn as nn
import torchvision.models as models


class RegressionResNet18(nn.Module):
    def __init__(self):
        super(RegressionResNet18, self).__init__()

        # Load the ResNet18 model
        self.resnet18 = models.resnet18(pretrained=False)

        # Remove the classification head
        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])

        # Add regression head
        self.regression_head = nn.Linear(
            512, 8
        )  # 512 is the output feature size from resnet18

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)
        return x


# Test the model
if __name__ == "__main__":
    import torch

    model = RegressionResNet18()
    image = torch.randn(1, 3, 256, 256)  # dummy image
    output = model(image)
    print(output.shape)  # should be torch.Size([1, 8])
