from mmcv.runner import auto_fp16

from mmdet.models import HEADS
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module()
class OlnFCNMaskHead(FCNMaskHead):

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        _x = x
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred, _x
