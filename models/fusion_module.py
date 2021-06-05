import torch
import torch.nn as nn
import full_image_colorization, instance_colorization

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

        self.instance_model = instance_colorization.InstanceColorization()
        # TODO: Set instance model to evaluation mode 
        self.instance_model.eval()

    def forward(self, input_list):
        '''
        Inputs:
            - input_A: grayscale full image, with shape (B, H, W) = (1, H, W)
            - instances: with shape (N, H, W), where N is the number of instances in the full image
            - bboxes_collection: a dict of format
                {
                    "32": box_info_32,
                    "64": box_info_64,
                    "128": box_info_128,
                    "256": box_info_256
                }
            box_info_32 has shape (N, 6), which stores bbox information of all N instances at 32x32 resolution. (padding_left, padding_right, padding_top, padding_bottom, rh, rw) = box_info_32[i] gives information of the i-th instance.
                - rh, rw: resized height and weight of the bounding box after mapping to the 32x32 map. Used by nn.functional.interpolate.
                - padding_left, padding_right, padding_top, padding_bottom: Used by nn.functional.pad to pad the resized weight map back to 32x32.
        Note:
            - Batch size should set to 1
        '''
        # Extract inputs
        input_A = input_list[0][0]
        instances = input_list[1][0]
        bboxes_collection = {key: item[0] for key, item in input_list[2].items()}

        # Extract features from instance images
        # instance_features_collection["key"] has shape (N, C', H', W')
        _, instance_features_collection = self.instance_model(instances)

        input_A = input_A.unsqueeze(1) # shape: (B, 1, H, W) = (1, 1, H, W)
        mask_1 = torch.zeros_like(input_A) # Placeholder, not used in this paper
        mask_2 = torch.zeros_like(input_A) # Placeholder, not used in this paper
        mask_3 = torch.zeros_like(input_A) # Placeholder, not used in this paper

        conv1 = self.model1(torch.cat((input_A, mask_1, mask_2, mask_3), dim=1))
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
        conv6 = self.fusion6(conv6, instance_features_collection["conv6"], bboxes_collection["32"])
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
        return out_reg

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
        '''
        Inputs:
            - full_image_feature: (1, C, H, W)
            - instance_features: (N, C, H, W)
            - bboxes: (N, 6)
        '''
        full_image_weight_map = self.full_image_convs(full_image_feature)
        # stacked_weight_map: (1, 1 + N, H, W)
        stacked_weight_map = full_image_weight_map.clone()
        resized_and_padded_feature_map_list = []
        for i, instance_feature in enumerate(instance_features):
            instance_feature = instance_feature.unsqueeze(0) # with shape (1, C, H, W)
            instance_weight_map = self.instance_convs(instance_feature)
            # bbox: [padding_left, padding_right, padding_top, padding_bottom, rh, rw]
            bbox = bboxes[i]
            # Resize the instance_feature and the instance_weight_map to respect the bbox, which defines the size and location of the instance at the full_image_feature.
            instance_feature = nn.functional.interpolate(instance_feature, size=(bbox[4], bbox[5]), mode='nearest')
            # Zero padding to the size of full_image_feature
            instance_feature = nn.functional.pad(instance_feature, (bbox[0], bbox[1], bbox[2], bbox[3]))
            instance_weight_map = nn.functional.interpolate(instance_weight_map, size=(bbox[4], bbox[5]), mode='nearest')
            instance_weight_map = nn.functional.pad(instance_weight_map, (bbox[0], bbox[1], bbox[2], bbox[3]), value=-100000)
            resized_and_padded_feature_map_list.append(instance_feature)

            torch.cat((stacked_weight_map, instance_weight_map), dim=1)
            # After that, we stack all the weight maps, apply softmax on each pixel, and obtain the fused feature using a weighted sum
        stacked_weight_map = self.softmax_norm(stacked_weight_map)
        # Broadcasting element-wise multiplication
        # TODO: could be wrong
        fused_feature = stacked_weight_map[:, 0, :, :].unsqueeze(1) * full_image_feature # with shape (1, C, H, W)
        for i in range(stacked_weight_map.shape[1] - 1):
            fused_feature += stacked_weight_map[:, i + 1, :, :].unsqueeze(1) * resized_and_padded_feature_map_list[i]
        return fused_feature # with shape (1, C, H, W)
