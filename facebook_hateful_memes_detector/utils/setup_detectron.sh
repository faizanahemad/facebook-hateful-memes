#! /bin/sh
# Look here: https://github.com/faizanahemad/ImageCaptioning.pytorch.git
# Look here: https://github.com/facebookresearch/vilbert-multi-task/blob/master/demo.ipynb


pip uninstall -y maskrcnn_benchmark
rm -rf vqa-maskrcnn-benchmark
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
#sed -i '/from maskrcnn_benchmark import _C/c\from ._utils import _C' maskrcnn_benchmark/layers/nms.py
#cat maskrcnn_benchmark/layers/nms.py
cd vqa-maskrcnn-benchmark

# %sed -i '/from maskrcnn_benchmark import _C/c\from ._utils import _C' layers/nms.py
# %cat layers/nms.py
python setup.py build
python setup.py develop
rm -rf detectron_model.pth
rm -rf detectron_model.yaml
wget -O detectron_model.pth wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget -O detectron_model.yaml wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml


pip install gdown
gdown --id 1VmUzgu0qlmCMqM1ajoOZxOXP3hiC_qlL
gdown --id 1zQe00W02veVYq-hdq5WsPOS3OPkNdq79
pip install yacs

pip uninstall -y captioning
pip install git+https://github.com/faizanahemad/ImageCaptioning.pytorch.git

pip install fvcore
pip install opencv-python
pip install 'git+https://github.com/faizanahemad/detectron2.git'
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop


