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

In addition, install [timm=0.4.9](https://github.com/rwightman/pytorch-image-models) for the Vision Transformer [(ViT)](https://arxiv.org/abs/2010.11929) models.

The code has been tested with CUDA 10.2/CuDNN 7.6.5 and PyTorch 1.9.0.

### Pre-Training

Similar to MoCo, only **multi-gpu**, **DistributedDataParallel** training is supported; single-gpu or DataParallel training is not supported. In addition, the code is improved to better suit the **multi-node** setting, and by default uses automatic **mixed-precision** for pre-training.

Below we list some MoCo v3 pre-training commands as examples. They cover different model architectures, training epochs, single-/multi-node, etc. 

<details>
<summary>ResNet-50, 100-Epoch, 2-Node.</summary>

This is the *default* setting for most hyper-parameters. With a batch size of 4096, the training fits into 2 nodes with a total of 16 Volta 32G GPUs. 

On the first node, run:
```
python main_moco.py \
  --dist-url 'tcp://[your node 1 address]:[specified port]'' \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run:
```
python main_moco.py \
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
  --optimizer=adamw --lr=1e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

Note that the smaller batch size: 1) facilitates stable training, as discussed in the [paper](https://arxiv.org/abs/2104.02057); and 2) cuts inter-node communication cost with single node training. Therefore, we highly recommend this setting for ViT-based explorations.

</details>

### Linear Classification

By default, we use SGD+Momentum optimizer and a batch size of 1024 for linear classification on frozen features/weights. This fits on an 8-GPU node.

<details>
<summary>Example linear classification command.</summary>

```
python main_lincls.py \
  -a [architecture] \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```
</details>

### Reference Setups

For longer pre-trainings with ResNet-50, we find the following hyper-parameters work well:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">epochs<br/></th>
<th valign="bottom">learning<br/>rate</th>
<th valign="bottom">weight<br/>decay</th>
<th valign="bottom">momentum<br/>update</th>
<th valign="center">top-1 acc.</th>
<!-- TABLE BODY -->
<tr>
<td align="center">100</td>
<td align="center">0.45</td>
<td align="center">1e-6</td>
<td align="center">0.99</td>
<td align="center"></td>
</tr>
<tr>
<td align="center">300</td>
<td align="center">0.3</td>
<td align="center">1e-6</td>
<td align="center">0.99</td>
<td align="center">72.8</td>
</tr>
<tr>
<td align="center">1000</td>
<td align="center">0.3</td>
<td align="center">1.5e-6</td>
<td align="center">0.996</td>
<td align="center">74.8</td>
</tr>
</tbody></table>

These hyper-parameters can be set with respective arguments. For example:

<details>
<summary>ResNet-50, 1000-Epoch, 2-Node.</summary>

On the first node, run:
```
python main_moco.py \
  --moco-m=0.996 --lr=.3 --wd=1.5e-6 --epochs=1000 \
  --dist-url "tcp://[your node 1 address]:[specified port]" \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run the same command as above, with `--rank 1`.
</details>

We also provide the reference linear classification performance in the last column (will update logs/pre-trained models soon).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.