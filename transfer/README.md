## MoCo v3 Transfer Learning with ViT

This folder includes the transfer learning experiments on CIFAR-10, CIFAR-100, Flowers and Pets datasets. We provide finetuning recipes for the ViT-Base model.

### Transfer Results

The following results are based on ImageNet-1k self-supervised pre-training, followed by end-to-end fine-tuning on downstream datasets. All results are based on a batch size of 128 and 100 training epochs.

#### ViT-Base, transfer learning
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">dataset</th>
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">pretrain<br/>crops</th>
<th valign="center">finetune<br/>epochs</th>
<th valign="center">transfer<br/>acc</th>
<!-- TABLE BODY -->
<tr>
<td align="left">CIFAR-10</td>
<td align="right">300</td>
<td align="center">2x224</td>
<td align="right">100</td>
<td align="center">98.9</td>
</tr>
<tr>
<td align="left">CIFAR-100</td>
<td align="right">300</td>
<td align="center">2x224</td>
<td align="right">100</td>
<td align="center">90.5</td>
</tr>
<tr>
<td align="left">Flowers</td>
<td align="right">300</td>
<td align="center">2x224</td>
<td align="right">100</td>
<td align="center">97.7</td>
</tr>
<tr>
<td align="left">Pets</td>
<td align="right">300</td>
<td align="center">2x224</td>
<td align="right">100</td>
<td align="center">93.2</td>
</tr>
</tbody></table>

Similar to the end-to-end fine-tuning experiment on ImageNet, the transfer learning results are also obtained using the [DEiT](https://github.com/facebookresearch/deit) repo, with the default model [deit_base_patch16_224]. 

### Preparation: Transfer learning with ViT

To perform transfer learning for ViT, use our script to convert the pre-trained ViT checkpoint to [DEiT](https://github.com/facebookresearch/deit) format:
```
python convert_to_deit.py \
  --input [your checkpoint path]/[your checkpoint file].pth.tar \
  --output [target checkpoint file].pth
```
Then copy (or replace) the following files to the DeiT folder: 
```
datasets.py
oxford_flowers_dataset.py
oxford_pets_dataset.py 
```

#### Download and prepare the datasets

Pets [\[Homepage\]](https://www.robots.ox.ac.uk/~vgg/data/pets/)
```
./data/
└── ./data/pets/
    ├── ./data/pets/annotations/               # split and label files
    └── ./data/pets/images/                    # data images
```

Flowers [\[Homepage\]](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
```
./data/
└── ./data/flowers/
    ├── ./data/flowers/jpg/               # jpg images
    ├── ./data/flowers/setid.mat          # dataset split   
    └── ./data/flowers/imagelabels.mat    # labels   
```


CIFAR-10/CIFAR-100 datasets will be downloaded automatically.


### Transfer learning scripts (with a 8-GPU machine):

#### CIFAR-10
```
python -u -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --batch-size 128 --output_dir [your output dir path] --epochs 100 --lr 3e-4 --weight-decay 0.1 \
    --no-pin-mem  --warmup-epochs 3 --data-set cifar10 --data-path [cifar-10 data path]  --no-repeated-aug \
    --resume [your pretrain checkpoint file] \
    --reprob 0.0 --drop-path 0.1 --mixup 0.8 --cutmix 1
```

#### CIFAR-100
```
python -u -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --batch-size 128 --output_dir [your output dir path] --epochs 100 --lr 3e-4 --weight-decay 0.1 \
    --no-pin-mem  --warmup-epochs 3 --data-set cifar100 --data-path [cifar-100 data path]  --no-repeated-aug \
    --resume [your pretrain checkpoint file] \
    --reprob 0.0 --drop-path 0.1 --mixup 0.5 --cutmix 1
```

#### Flowers
```
python -u -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --batch-size 128 --output_dir [your output dir path] --epochs 100 --lr 3e-4 --weight-decay 0.3 \
    --no-pin-mem  --warmup-epochs 3 --data-set flowers --data-path [oxford-flowers data path]  --no-repeated-aug \
    --resume [your pretrain checkpoint file] \
    --reprob 0.25 --drop-path 0.1 --mixup 0 --cutmix 0
```

#### Pets
```
python -u -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --batch-size 128 --output_dir [your output dir path] --epochs 100 --lr 3e-4 --weight-decay 0.1 \
    --no-pin-mem  --warmup-epochs 3 --data-set pets --data-path [oxford-pets data path]  --no-repeated-aug \
    --resume [your pretrain checkpoint file] \
    --reprob 0 --drop-path 0 --mixup 0.8 --cutmix 0
```

**Note**:
Similar to the ImageNet end-to-end finetuning experiment, we use `--resume` rather than `--finetune` in the DeiT repo, as its `--finetune` option trains under eval mode. When loading the pre-trained model, revise `model_without_ddp.load_state_dict(checkpoint['model'])` with `strict=False`.
