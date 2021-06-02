import torch
import torch.nn as nn


# Reference: https://github.com/junyanz/interactive-deep-colorization/

class FullImageColorization(nn.Module):
    def __init__(self):
        super(FullImageColorization, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model8up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        )

        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.model9up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        )

        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.model10up = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        )

        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=.2)
        )

        # Symmetric Shortcut connections
        self.model3short8 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.model2short9 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.model1short10 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1))

        # classification output
        self.model_class = nn.Sequential(nn.Conv2d(256, 529, kernel_size=1))

        self.model_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.Tanh()
        )

        self.upsample4 = nn.Sequential(nn.Upsample(scale_factor=4))
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, input_A):
        # input_A has shape (batch_size, H, W)
        # input_A \in [-50,+50]
        input_A = input_A.unsqueeze(1) # shape: (B, 1, H, W)
        mask_1 = torch.zeros_like(input_A) # Placeholder, not used in this paper
        mask_2 = torch.zeros_like(input_A) # Placeholder, not used in this paper
        mask_3 = torch.zeros_like(input_A) # Placeholder, not used in this paper

        conv1 = self.model1(torch.cat((input_A, mask_1, mask_2, mask_3), dim=1))
        # For conv2, conv3 and conv4, feature tensors are progressively halved spatially
        conv2 = self.model2(conv1[:, :, ::2, ::2])
        conv3 = self.model3(conv2[:, :, ::2, ::2])
        conv4 = self.model4(conv3[:, :, ::2, ::2])
        conv5 = self.model5(conv4)
        conv6 = self.model6(conv5)
        conv7 = self.model7(conv6)

        conv8_up = self.model8up(conv7) + self.model3short8(conv3)
        conv8 = self.model8(conv8_up)

        # TODO: not sure why returning out_cl
        out_cl = self.model_class(conv8)

        conv9_up = self.model9up(conv8) + self.model2short9(conv2)
        conv9 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9) + self.model1short10(conv1)
        conv10 = self.model10(conv10_up)
        out_reg = self.model_out(conv10)
        return out_reg
