#https://github.com/airsplay/py-bottom-up-attention/commit/c053e580da10da7e6639d3b26d2cc5b58207877a#diff-3578095dcbc32448d58a08f0ccbdaf91

from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsForVGStride(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        res5_halve        = cfg.MODEL.ROI_BOX_HEAD.RES5HALVE
        if not res5_halve:
            print("Modifications for VG in RoI heads (modeling/roi_heads/roi_heads.py):\n"
                  "\t1. Change the stride of conv1 and shortcut in Res5.Block1 from 2 to 1.\n"
                  "\t2. Modifying all conv2 with (padding: 1 --> 2) and (dilation: 1 --> 2).\n"
                  "\tFor more details, please check 'https://github.com/peteanderson80/bottom-up-attention/blob/master/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'.\n")
            self.res5[0].conv1.stride = (1, 1)
            self.res5[0].shortcut.stride = (1, 1)
            for i in range(3):
                self.res5[i].conv2.padding = (2, 2)
                self.res5[i].conv2.dilation = (2, 2)
