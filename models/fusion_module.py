import torch
import torch.nn as nn


class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )

        self.fusion1 = PerLayerFusion(64)

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.fusion2 = PerLayerFusion(128)

        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.fusion3 = PerLayerFusion(256)

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.fusion4 = PerLayerFusion(512)

        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.fusion5 = PerLayerFusion(512)

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.fusion6 = PerLayerFusion(512)

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.fusion7 = PerLayerFusion(512)

        self.model8up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        )

        self.fusion8up = PerLayerFusion(256)

        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.fusion8 = PerLayerFusion(256)

        self.model9up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        )

        self.fusion9up = PerLayerFusion(128)

        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.fusion9 = PerLayerFusion(128)

        self.model10up = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        )

        self.fusion10up = PerLayerFusion(128)

        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=.2)
        )

        self.fusion10 = PerLayerFusion(128)

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

    def forward(self, input_A, input_B, mask_B, instance_features_collection, bboxes_collection):
       # TODO: check the format of input 
        input_A = torch.Tensor(input_A).unsqueeze(0)
        input_B = torch.Tensor(input_B).unsqueeze(0)
        mask_B = torch.Tensor(mask_B).unsqueeze(0)

        conv1 = self.model1(torch.cat((input_A / 100., input_B, mask_B), dim=1))
        # inputs to PerLayerFusion: full_image_feature, instance_features, bboxes
        conv1 = self.fusion1(conv1, instance_features_collection["conv1"], bboxes_collection["256"])
        conv2 = self.model2(conv1[:, :, ::2, ::2])
        conv2 = self.fusion2(conv2, instance_features_collection["conv2"], bboxes_collection["128"])
        conv3 = self.model3(conv2[:, :, ::2, ::2])
        conv3 = self.fusion3(conv3, instance_features_collection["conv3"], bboxes_collection["64"])
        conv4 = self.model4(conv3[:, :, ::2, ::2])
        conv4 = self.fusion4(conv4, instance_features_collection["conv4"], bboxes_collection["32"])
        conv5 = self.model5(conv4)
        conv5 = self.fusion5(conv5, instance_features_collection["conv5"], bboxes_collection["32"])
        conv6 = self.model6(conv5)
        conv6 = self.fusion1(conv6, instance_features_collection["conv6"], bboxes_collection["32"])
        conv7 = self.model7(conv6)
        conv7 = self.fusion7(conv7, instance_features_collection["conv7"], bboxes_collection["32"])

        conv8_up = self.model8up(conv7) + self.model3short8(conv3)
        conv8_up = self.fusion8up(conv8_up, instance_features_collection["conv8_up"], bboxes_collection["64"])
        conv8 = self.model8(conv8_up)
        conv8 = self.fusion8(conv8, instance_features_collection["conv8"], bboxes_collection["64"])

        conv9_up = self.model9up(conv8) + self.model2short9(conv2)
        conv9_up = self.fusion9up(conv9_up, instance_features_collection["conv9_up"], bboxes_collection["128"])
        conv9 = self.model9(conv9_up)
        conv9 = self.fusion9(conv9, instance_features_collection["conv9"], bboxes_collection["128"])
        conv10_up = self.model10up(conv9) + self.model1short10(conv1)
        conv10_up = self.fusion10up(conv10_up, instance_features_collection["conv10_up"], bboxes_collection["256"])
        conv10 = self.model10(conv10_up)
        conv10 = self.fusion10(conv10, instance_features_collection["conv10"], bboxes_collection["256"])
        out_reg = self.model_out(conv10)
        return out_reg * 110

class PerLayerFusion(nn.Module):
    def __init__(self, in_channels):
        super(PerLayerFusion, self).__init__()
        
        self.full_image_convs = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        self.instance_convs = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        self.softmax_norm = nn.Softmax(dim=1)

    def forward(self, full_image_feature, instance_features, bboxes):
        # full_image_feature: (1, C, H, W)
        full_image_weight_map = self.full_image_conv(full_image_feature)
        # stacked_weight_map: (1, C', H, W)
        stacked_weight_map = full_image_weight_map.clone()
        resized_and_padded_feature_map_list = []
        for i, instance_feature in enumerate(instance_features):
            instance_weight_map = self.instance_convs(instance_feature)
            # bbox: [L_pad, R_pad, T_pad, B_pad, rh, rw]
            bbox = bboxes[i]
            # Resize the instance_feature and the instance_weight_map to respect the bbox, which defines the size and location of the instance at the full_image_feature. 
            instance_feature = nn.functional.interpolate(instance_feature, size=(bbox[4], bbox[5]), mode='nearest')
            # Zero padding to the size of full_image_feature
            instance_feature = nn.functional.pad(instance_feature, (bbox[0], bbox[1], bbox[2], bbox[3]))
            instance_weight_map = nn.functional.interpolate(instance_weight_map, size=(bbox[4], bbox[5]), mode='nearest')
            instance_weight_map = nn.functional.pad(instance_weight_map, (bbox[0], bbox[1], bbox[2], bbox[3]))
            resized_and_padded_feature_map_list.append(instance_feature)

            torch.cat((stacked_weight_map, instance_weight_map), dim=1)
            # After that, we stack all the weight maps, apply softmax on each pixel, and obtain the fused feature using a weighted sum
        stacked_weight_map = self.softmax_norm(stacked_weight_map)
        fused_feature = stacked_weight_map[:, 0, :, :] * full_image_feature
        for i in range(stacked_weight_map.size[1] - 1):
            fused_feature += stacked_weight_map[:, i + 1, :, :] * resized_and_padded_feature_map_list[i]
        return fused_feature