#https://github.com/airsplay/py-bottom-up-attention/commit/c053e580da10da7e6639d3b26d2cc5b58207877a#diff-ba16c545429f1c96b5ee0248c1d56469

# Caffe Options
_C.MODEL.CAFFE_MAXPOOL = False
_C.MODEL.PROPOSAL_GENERATOR.HID_CHANNELS = -1
del _C.MODEL.ANCHOR_GENERATOR.OFFSET

# RPN options
_C.MODEL.ROI_BOX_HEAD.RES5HALVE = True

del _C.VIS_PERIOD