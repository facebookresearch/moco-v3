# MoCo v3

This is a PyTorch implementation of [MoCo v3](https://arxiv.org/abs/2104.02057):
```
@Article{chen2021mocov3,
  author  = {Xinlei Chen* and Saining Xie* and Kaiming He},
  title   = {An Empirical Study of Training Self-Supervised Vision Transformers},
  journal = {arXiv preprint arXiv:2104.02057},
  year    = {2021},
}
```

### Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo](https://github.com/facebookresearch/moco), the code release contains minimal modifications for both unsupervised pre-training and linear classification to that code. 

In addition, install [timm](https://github.com/rwightman/pytorch-image-models) for the Vision Transformer [(ViT)](https://arxiv.org/abs/2010.11929) models.

The code has been tested with CUDA 10.2/CuDNN 7.6.5, PyTorch 1.9.0 and timm 0.4.9.

### Pre-Training

Similar to MoCo, only **multi-gpu**, **DistributedDataParallel** training is supported; single-gpu or DataParallel training is not supported. In addition, the code is improved to better suit the **multi-node** setting, and by default uses automatic **mixed-precision** for pre-training.

Below we list some MoCo v3 pre-training commands as examples. They cover different architectures, training epochs, single-/multi-node training, etc. 

<details>
<summary>ResNet-50, 100-Epoch, 2-Node.</summary>

This is the *default* setting for most hyper-parameters. With a batch size of 4096, the training fits into 2 nodes with a total of 16 Volta 32G GPUs. 

On the first node, run:
```
python main_moco.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://[your node 1 address]:[specified port]'' \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run:
```
python main_moco.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://[your node 1 address]:[specified port]' \
  --multiprocessing-distributed --world-size 2 --rank 1 \
  [your imagenet-folder with train and val folders]
```
</details>

<details>
<summary>ViT-Small, 300-Epoch, 1-Node.</summary>

With a batch size of 1024, ViT-Small fits into a single node of 8 Volta 32G GPUs:

```
python main_moco.py \
  -a vit_small -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
</details>

Note that the smaller batch size: 1) facilitates stable training, as discussed in the [paper](https://arxiv.org/abs/2104.02057); and 2) cuts inter-node communication cost with single node training. Therefore, we recommend this setting for ViT-based explorations.

### Linear Classification

By default, we use SGD+Momentum optimizer and a batch size of 1024 for linear classification on frozen features/weights. This fits on an 8-GPU node.

<details>
<summary>Linear classification command.</summary>

```
python main_lincls.py \
  -a [architecture] --lr [learning rate] \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```
</details>

### End-to-End Classification

To perform end-to-end fine-tuning for ImageNet classification, first convert the pre-trained checkpoints to [DEiT](https://github.com/facebookresearch/deit) format:
```
python convert_to_deit.py \
  --input [your checkpoint path]/[your checkpoint file].pth.tar \
  --output [target checkpoint file].pth
```
Then use `[target checkpoint file].pth` to initialize weights in DEiT.

With 100-epoch fine-tuning, the reference top-1 classification accuracy is 82.8%. With 300-epoch, the accuracy is 83.2%.

### Reference Setups and Models

For longer pre-trainings with ResNet-50, we find the following hyper-parameters work well (expected performance in the last column):

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">epochs</th>
<th valign="bottom">learning<br/>rate</th>
<th valign="bottom">weight<br/>decay</th>
<th valign="bottom">momentum<br/>update</th>
<th valign="bottom">momentum<br/>schedule</th>
<th valign="bottom">crop<br/>min-scale</th>
<th valign="center">top-1 acc.</th>
<!-- TABLE BODY -->
<tr>
<td align="center">100</td>
<td align="center">0.6</td>
<td align="center">1e-6</td>
<td align="center">0.99</td>
<td align="center">cosine</td>
<td align="center">0.2</td>
<td align="center">69.0</td>
</tr>
<tr>
<td align="center">300</td>
<td align="center">0.3</td>
<td align="center">1e-6</td>
<td align="center">0.99</td>
<td align="center">cosine</td>
<td align="center">0.2</td>
<td align="center">73.0</td>
</tr>
<tr>
<td align="center">1000</td>
<td align="center">0.3</td>
<td align="center">1.5e-6</td>
<td align="center">0.996</td>
<td align="center">cosine</td>
<td align="center">0.2</td>
<td align="center">[TODO]74.8</td>
</tr>
</tbody></table>

These hyper-parameters can be set with respective arguments. For example:

<details>
<summary>MoCo v3 with ResNet-50, 1000-Epoch Training.</summary>

On the first node, run:
```
python main_moco.py \
  --lr=.3 --wd=1.5e-6 --epochs=1000 \
  --moco-m=0.996 --moco-m-cos --crop-min=.2 \
  --dist-url "tcp://[your node 1 address]:[specified port]" \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run the same command as above, with `--rank 1`.
</details>

For ViT, we find the following hyper-parameters work well (different from ResNet-50):

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">model<br/>size</th>
<th valign="center">optimizer</th>
<th valign="bottom">learning<br/>rate</th>
<th valign="bottom">weight<br/>decay</th>
<th valign="bottom">warmup<br/>epochs</th>
<th valign="center">temperature</th>
<th valign="center">top-1 acc.</th>
<!-- TABLE BODY -->
<tr>
<td align="center">ViT-Small</td>
<td align="center">AdamW</td>
<td align="center">1.5e-4</td>
<td align="center">0.1</td>
<td align="center">40</td>
<td align="center">0.2</td>
<td align="center">[TODO]73.1</td>
</tr>
<tr>
<td align="center">ViT-Base</td>
<td align="center">AdamW</td>
<td align="center">1.5e-4</td>
<td align="center">0.1</td>
<td align="center">40</td>
<td align="center">0.2</td>
<td align="center">76.7</td>
</tr>
</tbody></table>

And for large batch size training, it is especially important to set `--stop-grad-conv1` so the first layer is a fixed random patch projection. For example:

<details>
<summary>ViT-Base, 300-Epoch, 8-Node.</summary>

With a batch size of 4096, ViT-Base can be trained on 8 nodes:

```
python main_moco.py \
  -a vit_base \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-t=.2 \
  --dist-url 'tcp://[your node 1 address]:[specified port]'' \
  --multiprocessing-distributed --world-size 8 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the other nodes, run the same command as above, with `--rank [r]` where `[r]` ranges from integers `1` to `7`.
</details> 

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.