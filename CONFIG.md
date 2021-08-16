## MoCo v3 Reference Setups and Models

Here we document the reference commands for pre-training and evaluating various MoCo v3 models.

### ResNet-50 models

With batch 4096, the training of all ResNet-50 models can fit into 2 nodes with a total of 16 Volta 32G GPUs. 

<details>
<summary>ResNet-50, 100-epoch pre-training.</summary>

On the first node, run:
```
python main_moco.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://[your first node address]:[specified port]' \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run the same command with `--rank 1`.
</details>

<details>
<summary>ResNet-50, 300-epoch pre-training.</summary>

On the first node, run:
```
python main_moco.py \
  --lr=.3 --epochs=300 \
  --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://[your first node address]:[specified port]' \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run the same command with `--rank 1`.
</details>

<details>
<summary>ResNet-50, 1000-epoch pre-training.</summary>

On the first node, run:
```
python main_moco.py \
  --lr=.3 --wd=1.5e-6 --epochs=1000 \
  --moco-m=0.996 --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://[your first node address]:[specified port]' \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On the second node, run the same command with `--rank 1`.
</details>

<details>
<summary>ResNet-50, linear classification.</summary>

Run on single node:
```
python main_lincls.py \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```
</details>

Below are our pre-trained ResNet-50 models and logs.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">linear<br/>acc</th>
<th valign="center">pretrain<br/>files</th>
<th valign="center">linear<br/>files</th>
<!-- TABLE BODY -->
<tr>
<td align="right">100</td>
<td align="center">68.9</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/r-50-100ep.pth.tar">chpt</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/linear-100ep.pth.tar">chpt</a> /
                  <a href="https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/linear-100ep.std">log</a></td>
</tr>
<tr>
<td align="right">300</td>
<td align="center">72.8</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/r-50-300ep.pth.tar">chpt</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/linear-300ep.pth.tar">chpt</a> /
                  <a href="https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/linear-300ep.std">log</a></td>
</tr>
<tr>
<td align="right">1000</td>
<td align="center">74.6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar">chpt</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar">chpt</a> /
                  <a href="https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.std">log</a></td>
</tr>
</tbody></table>


### ViT Models

All ViT models are pre-trained for 300 epochs with AdamW.

<details>
<summary>ViT-Small, 1-node (8-GPU), 1024-batch pre-training.</summary>

This setup fits into a single node of 8 Volta 32G GPUs, for ease of debugging.
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

</details>

<details>
<summary>ViT-Small, 4-node (32-GPU) pre-training.</summary>

On the first node, run:
```
python main_moco.py \
  -a vit_small \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://[your first node address]:[specified port]' \
  --multiprocessing-distributed --world-size 8 --rank 0 \
  [your imagenet-folder with train and val folders]
```
On other nodes, run the same command with `--rank 1`, ..., `--rank 3` respectively.
</details>

<details>
<summary>ViT-Small, linear classification.</summary>

Run on single node:
```
python main_lincls.py \
  -a vit_small --lr=3 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```
</details>

<details>
<summary>ViT-Base, 8-node (64-GPU) pre-training.</summary>

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
</details>

<details>
<summary>ViT-Base, linear classification.</summary>

Run on single node:
```
python main_lincls.py \
  -a vit_base --lr=3 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```
</details>


Below are our pre-trained ViT models and logs (batch 4096).

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">model</th>
<th valign="center">pretrain<br/>epochs</th>
<th valign="center">linear<br/>acc</th>
<th valign="center">pretrain<br/>files</th>
<th valign="center">linear<br/>files</th>
<!-- TABLE BODY -->
<tr>
<td align="left">ViT-Small</td>
<td align="center">300</td>
<td align="center">73.2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar">chpt</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/linear-vit-s-300ep.pth.tar">chpt</a> /
                  <a href="https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/linear-vit-s-300ep.std">log</a></td>
</tr>
<tr>
<td align="left">ViT-Base</td>
<td align="center">300</td>
<td align="center">76.7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar">chpt</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar">chpt</a> /
                  <a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.std">log</a></td>
</tr>
</tbody></table>
