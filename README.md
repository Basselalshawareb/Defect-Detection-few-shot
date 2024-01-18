##installation


pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmcv-full==1.4.6
pip install mmcls==0.16
pip install tensorboard
cd ./packages

cd ./mmdetection
pip install -r requirements.txt
pip install -e .

cd ../mmfewshot
pip install -r requirements.txt
pip install -e .

cd ../mmrazor
pip install -r requirements.txt
pip install -e .

cd ../.. lines 6 and 7
go to /home/huemorgen/miniconda3/envs/defect/lib/python3.9/site-packages/torch/utils/tensorboard/__init__.py
comment the following lines:
# if not hasattr(tensorboard, '__version__') or LooseVersion(tensorboard.__version__) < LooseVersion('1.15'):
#     raise ImportError('TensorBoard logging requires TensorBoard version 1.15 or above')

To use Jupiter Notebook install the following package:
conda install -n defect ipykernel --update-deps --force-reinstall -y

In case pretrained ResNet cannot be found, copy the weights from folder ResNet weights and paste it in the hidden directory ~/.torch/models
