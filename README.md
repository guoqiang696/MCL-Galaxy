# MCL-Galaxy

1. Pre-training model

   Use the “main_moco.py”  file in the folder, at least 4 Gpus, using the following instructions:

   

 python main_moco.py \

-a resnet50 \
  --lr 0.015 \
  --epochs 801 \
  --batch-size 128 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  guoqiang_train

(The  'guoqiang_train'  contains images of galaxy morphology)



2.pre-trained model

pre-trained model are Available for download via link, 200 and 800 epochs were trained respectively



3.downstream classification task

Put pre-trained model  and "galaxyzoo.py" in the same place and run the galaxyzoo.py file