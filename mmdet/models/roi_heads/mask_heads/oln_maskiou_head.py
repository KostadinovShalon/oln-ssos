from mmcv.cnn import Conv2d, Linear
from torch import nn

from mmdet.models.builder import HEADS
from .maskiou_head import MaskIoUHead
from torch.nn.modules.utils import _pair


@HEADS.register_module()
class OlnMaskIoUHead(MaskIoUHead):

    def __init__(self,
                 num_convs=1,
                 *args, **kwargs):
        super(OlnMaskIoUHead, self).__init__(*args, **kwargs)
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = self.conv_out_channels
            self.convs.append(
                Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    stride=1,
                    padding=1))
        # roi_feat_size = _pair(roi_feat_size)
        # pooled_area = (roi_feat_size[0] // 2) * (roi_feat_size[1] // 2)
        # self.fcs = nn.ModuleList()
        # for i in range(num_fcs):
        #     in_channels = (
        #         self.conv_out_channels *
        #         pooled_area if i == 0 else self.fc_out_channels)
        #     self.fcs.append(Linear(in_channels, self.fc_out_channels))

    def forward(self, mask_feat, mask_pred):
        x = mask_feat
        for conv in self.convs:
            x = self.relu(conv(x))
        x = self.max_pool(x)
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        return mask_iou
