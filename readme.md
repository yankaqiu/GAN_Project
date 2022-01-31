# 基于 CycleGAN 和 Stylegan2 网络模型在图像转换/生成的优化微调



## 实验环境

#### 实验平台：

系统版本：Ubuntu 16.04.6 LTS

显卡：NVIDIA Tesla V100 * 1

##### CycleGAN 相关实验：

Python 3.5

Pytorch: 1.10.1

##### Stylegan 相关实验：

64-bit Python 3.6

TensorFlow-gpu 1.15

Pytorch: 1.10.1



## 数据集下载

Simpsons: https://www.kaggle.com/kostastokis/simpsons-faces

FFHQ(Flickr-Faces-HQ):  https://github.com/NVlabs/ffhq-dataset)

horse2zebra dataset:https://github.com/togheppi/CycleGAN#:~:text=Tensorboard%20in%20PyTorch.-,horse2zebra%20dataset,-Image%20size%3A%20256x256



## 运行方式

#### CycleGAN-pytorch:

##### 1. Setup the dataset

可以下载或自己按如下文件结构构造数据集:

```
├── datasets                   
|   ├── <dataset_name>         # i.e. Humans2Simpsons
|   |   ├── train              # Training
|   |   |   ├── A              # Contains domain A images (i.e. Humans)
|   |   |   └── B              # Contains domain B images (i.e. Simpsons)
|   |   └── test               # Testing
|   |   |   ├── A              # Contains domain A images (i.e. Humans)
|   |   |   └── B              # Contains domain B images (i.e. Simpsons)
```

##### 2. Train!

```
./train --dataroot datasets/<dataset_name>/ --cuda
```

可以使用`./train --help` 查看可设置的训练参数。



#### WCycleGAN-pytorch/WCycleGAN-improved:

- 下载官方CycleGAN dataset (e.g. maps):

```
bash ./datasets/download_cyclegan_dataset.sh maps
```

- Train a model:

```
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --no_dropout
```

- 运行 `python -m visdom.server` 然后点击 URL [http://localhost:8097](http://localhost:8097/). 可以看到训练过程的 图片和loss函数图像
- Test the model:

```
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout
```

The test results will be saved to a html file here: 	`./results/maps_cyclegan/latest_test/index.html`.



#### StyleGAN2:

**1. Prepare LMDB Dataset**

First create lmdb datasets:

```
python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH
```

**2. Train**

```
python train.py --batch BATCH_SIZE LMDB_PATH
# ex) python train.py --batch=8 --ckpt=xx.pt --path=LMDB_PATH
```

**Options**

1. Project images to latent spaces

   ```
   python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...
   ```

2. [Closed-Form Factorization](https://arxiv.org/abs/2007.06600)

   You can use `closed_form_factorization.py` and `apply_factor.py` to discover meaningful latent semantic factor or directions in unsupervised manner.

   First, you need to extract eigenvectors of weight matrices using `closed_form_factorization.py`

   ```
   python closed_form_factorization.py [CHECKPOINT]
   ```

   This will create factor file that contains eigenvectors. (Default: factor.pt) And you can use `apply_factor.py` to test the meaning of extracted directions

   ```
   python apply_factor.py -i [INDEX_OF_EIGENVECTOR] -d [DEGREE_OF_MOVE] -n [NUMBER_OF_SAMPLES] --ckpt [CHECKPOINT] [FACTOR_FILE]
   # ex) python apply_factor.py -i 19 -d 5 -n 10 --ckpt [CHECKPOINT] factor.pt
   ```



## 实验结果

#### CycleGAN-pytorch:

第一列: Input / 第二列: Generated / 第三列: Reconstructed

Horses:

[<img src="https://s4.ax1x.com/2022/01/31/HigArn.png" alt="HigArn.png" style="zoom:67%;" />](https://imgtu.com/i/HigArn)

Simpsons:

[<img src="https://s4.ax1x.com/2022/01/31/HiczUf.png" alt="HiczUf.png" style="zoom:67%;" />](https://imgtu.com/i/HiczUf)

[<img src="https://s4.ax1x.com/2022/01/31/HigC8g.png" alt="HigC8g.png" style="zoom:67%;" />](https://imgtu.com/i/HigC8g)



Loss图像：

[<img src="https://s4.ax1x.com/2022/01/31/Higivj.png" alt="Higivj.png" style="zoom:67%;" />](https://imgtu.com/i/Higivj)



#### WCycleGAN-pytorch(improved):

第一列: Input / 第二列: Generated / 第三列: Reconstructed

Houses:

[<img src="https://s4.ax1x.com/2022/01/31/HigEbq.png" alt="HigEbq.png" style="zoom:67%;" />](https://imgtu.com/i/HigEbq)

Simpsons:

[<img src="https://s4.ax1x.com/2022/01/31/Hig9PS.png" alt="Hig9PS.png" style="zoom:67%;" />](https://imgtu.com/i/Hig9PS)

[<img src="https://s4.ax1x.com/2022/01/31/HigP2Q.png" alt="HigP2Q.png" style="zoom:67%;" />](https://imgtu.com/i/HigP2Q)

Loss 图像：

[<img src="https://s4.ax1x.com/2022/01/31/HigkKs.png" alt="HigkKs.png" style="zoom:67%;" />](https://imgtu.com/i/HigkKs)

三种方法在 Simpsons faces<->人脸 任务对比结果：

| Loss               | Per-pixel ace. | Per-class acc. | Class IOU |
| ------------------ | -------------- | -------------- | --------- |
| CycleGAN           | 0.22           | 0.07           | 0.02      |
| WCycleGAN          | 0.34           | 0.13           | 0.10      |
| WCycleGAN-improved | 0.42           | 0.22           | 0.09      |



#### StyleGAN2-Freeze

基于 FFHQ 的预训练集在 Simpsons上训练结果：

[![HigS58.png](https://s4.ax1x.com/2022/01/31/HigS58.png)](https://imgtu.com/i/HigS58)

详细结果及分析见报告