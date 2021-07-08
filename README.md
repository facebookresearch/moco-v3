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

### Unsupervised Pre-Training

Similar to MoCo, only **multi-gpu**, **DistributedDataParallel** training is supported; single-gpu or DataParallel training is not supported. In addition, the code is tested with **multi-node** setting, and by default uses automatic **mixed-precision** for pre-training.

Below we exemplify several pre-training commands covering different model architectures, training epochs, single-/multi-node, etc. 

<details>
<summary>
MoCo v3 with ResNet-50, 100-Epoch, 2-Node.
</summary>

This is the *default* setting for most hyper-parameters. With a batch size of 4096, the training fits into 2 nodes with a total of 16 Volta 32G GPUs. 

On the first node, run:
```
python main_moco.py \
  --dist-url "tcp://[your node 1 address]:[specified port]" \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run:
```
python main_moco.py \
  --dist-url "tcp://[your node 1 address]:[specified port]" \
  --multiprocessing-distributed --world-size 2 --rank 1 \
  [your imagenet-folder with train and val folders]
```
</details>

<details>
<summary>
MoCo v3 with ResNet-50, 300-Epoch, 2-Node.
</summary>

On the first node, run:
```
python main_moco.py \
  --lr=.3 --epochs=300 \
  --dist-url "tcp://[your node 1 address]:[specified port]" \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run the same command as above, with `--rank 1`.
</details>

<details>
<summary>
MoCo v3 with ResNet-50, 1000-Epoch, 2-Node.
</summary>

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

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.