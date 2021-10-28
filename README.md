## MoCo v3 for Self-supervised ResNet and ViT

### Introduction
This is a PyTorch implementation of [MoCo v3](https://arxiv.org/abs/2104.02057) for self-supervised ResNet and ViT.

The original MoCo v3 was implemented in Tensorflow and run in TPUs. This repo re-implements in PyTorch and GPUs. Despite the library and numerical differences, this repo reproduces the results and observations in the paper. 

### Main Results

The following results are based on ImageNet-1k self-supervised pre-training, followed by ImageNet-1k supervised training for linear evaluation or end-to-end fine-tuning. All results in these tables are based on a batch size of 4096.

**Pre-trained models** and **configs** can be found at [CONFIG.md](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md).

#### ResNet-50, linear classification
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">pretrain<br/>crops</th>
<th valign="center">linear<br/>acc</th>
<!-- TABLE BODY -->
<tr>
<td align="right">100</td>
<td align="center">2x224</td>
<td align="center">68.9</td>
</tr>
<tr>
<td align="right">300</td>
<td align="center">2x224</td>
<td align="center">72.8</td>
</tr>
<tr>
<td align="right">1000</td>
<td align="center">2x224</td>
<td align="center">74.6</td>
</tr>
</tbody></table>

#### ViT, linear classification
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">model</th>
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">pretrain<br/>crops</th>
<th valign="center">linear<br/>acc</th>
<!-- TABLE BODY -->
<tr>
<td align="left">ViT-Small</td>
<td align="right">300</td>
<td align="center">2x224</td>
<td align="center">73.2</td>
</tr>
<tr>
<td align="left">ViT-Base</td>
<td align="right">300</td>
<td align="center">2x224</td>
<td align="center">76.7</td>
</tr>
</tbody></table>

#### ViT, end-to-end fine-tuning
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">model</th>
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">pretrain<br/>crops</th>
<th valign="center">e2e<br/>acc</th>
<!-- TABLE BODY -->
<tr>
<td align="left">ViT-Small</td>
<td align="right">300</td>
<td align="center">2x224</td>
<td align="center">81.4</td>
</tr>
<tr>
<td align="left">ViT-Base</td>
<td align="right">300</td>
<td align="center">2x224</td>
<td align="center">83.2</td>
</tr>
</tbody></table>

The end-to-end fine-tuning results are obtained using the [DeiT](https://github.com/facebookresearch/deit) repo, using all the default DeiT configs. ViT-B is fine-tuned for 150 epochs (vs DeiT-B's 300ep, which has 81.8% accuracy).

### Usage: Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo v1/2](https://github.com/facebookresearch/moco), this repo contains minimal modifications on the official PyTorch ImageNet code. We assume the user can successfully run the official PyTorch ImageNet code.
For ViT models, install [timm](https://github.com/rwightman/pytorch-image-models) (`timm==0.4.9`).

The code has been tested with CUDA 10.2/CuDNN 7.6.5, PyTorch 1.9.0 and timm 0.4.9.

### Usage: Self-supervised Pre-Training

Below are three examples for MoCo v3 pre-training. 

#### ResNet-50 with 2-node (16-GPU) training, batch 4096

On the first node, run:
```
python main_moco.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://[your first node address]:[specified port]' \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run the same command with `--rank 1`.
With a batch size of 4096, the training can fit into 2 nodes with a total of 16 Volta 32G GPUs. 


#### ViT-Small with 1-node (8-GPU) training, batch 1024

```
python main_moco.py \
  -a vit_small -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

#### ViT-Base with 8-node training, batch 4096

With a batch size of 4096, ViT-Base is trained with 8 nodes:
```
python main_moco.py \
  -a vit_base \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://[your first node address]:[specified port]' \
  --multiprocessing-distributed --world-size 8 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On other nodes, run the same command with `--rank 1`, ..., `--rank 7` respectively.

#### Notes:
1. The batch size specified by `-b` is the total batch size across all GPUs.
1. The learning rate specified by `--lr` is the *base* lr, and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677) in [this line](https://github.com/facebookresearch/moco-v3/blob/main/main_moco.py#L213).
1. Using a smaller batch size has a more stable result (see paper), but has lower speed. Using a large batch size is critical for good speed in TPUs (as we did in the paper).
1. In this repo, only *multi-gpu*, *DistributedDataParallel* training is supported; single-gpu or DataParallel training is not supported. This code is improved to better suit the *multi-node* setting, and by default uses automatic *mixed-precision* for pre-training.

### Usage: Linear Classification

By default, we use momentum-SGD and a batch size of 1024 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

```
python main_lincls.py \
  -a [architecture] --lr [learning rate] \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```

### Usage: End-to-End Fine-tuning ViT

To perform end-to-end fine-tuning for ViT, use our script to convert the pre-trained ViT checkpoint to [DEiT](https://github.com/facebookresearch/deit) format:
```
python convert_to_deit.py \
  --input [your checkpoint path]/[your checkpoint file].pth.tar \
  --output [target checkpoint file].pth
```
Then run the training (in the DeiT repo) with the converted checkpoint:
```
python $DEIT_DIR/main.py \
  --resume [target checkpoint file].pth \
  --epochs 150
```
This gives us 83.2% accuracy for ViT-Base with 150-epoch fine-tuning.

**Note**:
1. We use `--resume` rather than `--finetune` in the DeiT repo, as its `--finetune` option trains under eval mode. When loading the pre-trained model, revise `model_without_ddp.load_state_dict(checkpoint['model'])` with `strict=False`.
1. Our ViT-Small is with `heads=12` in the Transformer block, while by default in DeiT it is `heads=6`. Please modify the DeiT code accordingly when fine-tuning our ViT-Small model. 

### Model Configs

See the commands listed in [CONFIG.md](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md) for specific model configs, including our recommended hyper-parameters and pre-trained reference models.

### Transfer Learning

See the instructions in the [transfer](https://github.com/facebookresearch/moco-v3/tree/main/transfer) dir.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citation
```
@Article{chen2021mocov3,
  author  = {Xinlei Chen* and Saining Xie* and Kaiming He},
  title   = {An Empirical Study of Training Self-Supervised Vision Transformers},
  journal = {arXiv preprint arXiv:2104.02057},
  year    = {2021},
}
```
