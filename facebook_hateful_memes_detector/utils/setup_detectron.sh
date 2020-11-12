#! /bin/sh
# Look here: https://github.com/faizanahemad/ImageCaptioning.pytorch.git
# Look here: https://github.com/facebookresearch/vilbert-multi-task/blob/master/demo.ipynb
# Detection vocab of faster-rcnn model is of visual genome and is here: https://github.com/peteanderson80/bottom-up-attention/blob/master/data/genome/1600-400-20/objects_vocab.txt
# MS COCO labels: https://gist.githubusercontent.com/AruniRC/7b3dadd004da04c80198557db5da4bda/raw/2f10965ace1e36c4a9dca76ead19b744f5eb7e88/ms_coco_classnames.txt


#sed -i '/from maskrcnn_benchmark import _C/c\from ._utils import _C' maskrcnn_benchmark/layers/nms.py
#cat maskrcnn_benchmark/layers/nms.py
# %sed -i '/from maskrcnn_benchmark import _C/c\from ._utils import _C' layers/nms.py
# %cat layers/nms.py

rm -rf detectron_model.pth
rm -rf detectron_model.yaml
wget -O detectron_model.pth https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget -O detectron_model.yaml https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml

# https://github.com/airsplay/lxmert/issues/32
gdown --id 1VmUzgu0qlmCMqM1ajoOZxOXP3hiC_qlL
gdown --id 1zQe00W02veVYq-hdq5WsPOS3OPkNdq79


# https://github.com/ox-vgg/vgg_face2
wget -O senet50_256_pytorch.tar.gz http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/senet50_256_pytorch.tar.gz
tar -xvf senet50_256_pytorch.tar.gz

wget -O resnet50_256_pytorch.tar.gz http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/resnet50_256_pytorch.tar.gz
tar -xvf resnet50_256_pytorch.tar.gz resnet50_256.pth
