python main_moco.py \
  -a resnet18 \
  --optimizer=adamw \
  --lr=0.3 \
  --weight-decay=.1 \
  --epochs=100 \
  --warmup-epochs=25 \
  --batch-size 256 \
  --moco-t 0.2 \
  --dist-url 'tcp://localhost:10001' \
  --gpus 0,1,2,3 \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  ../../data/imagenet-100/