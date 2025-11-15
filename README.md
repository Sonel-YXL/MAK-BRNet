# MAK-BRNet: Multi-scale Adaptive Kernel and Boundary Refinement Network for Remote Sensing Object Detection

目前根据现有的公开数据集**DOTA**和**HRSC 2016**进行实验估计。Readme文件跳转[Configs](Configs/README.md),[Dataset](Datasets/README.md),[Models](Models/README.md),
[Tools](Tools/README.md),[Utils](Utils/README.md)


* [文件夹架构](
- [如何开始](
    - [环境构建](
    - [数据集](
    - [训练流程](
    - [验证流程](
    - [常用命令](
* [参考资料](
   - [参考代码网页](
   - [参考论文](
   - [参考论文改进流程](
- [致谢](



```
${SOTA-HTD}                     //根目录
├── Config 
│   ├── base                    //基类文件继承
│   │   ├── hrsc.yaml           //HRSC 2016 数据集配置文件
│   │   ├── soda.yaml           //SODA-a 数据集配置文件
│   │   ├── dota.yaml           //DOTA 数据集配置文件
│   ├── (数据集)_(模型名称)_(模块名称)_(数据集或文件标识)_(训练轮次).yaml
│   ├── HRSC_DBCNet_r50_fpn_Tvt_100x.yaml   // 使用hrsc数据集,DBCNet(r50主干网络和fpn),训练+验证数据训练100*12个epoch
│   ├── SODA_DBCNet_r50_fpn_tiny_6x.yaml    // 使用SODA数据集,DBCNet(r50主干网络和fpn),tiny(跑通)数据训练6*12个epoch
│   └── ...
├── Dataset                         //数据集处理代码存放目录(也可存放数据集,需修改相关路径)
│   ├── Data_OOD                    //数据集加载类,进行数据集管理
│   │   ├── OOD_dataset.py          //数据集加载类
│   │   └── OOD_datasetmanager.py   //数据集管理类
│   ├── Transform_OOD               //数据集预处理,数据增强,标签制作
│   ├── Format_Dataset              //对公有数据集的格式转换
│   ├── raw                         //原始数据集
│   ├── DOTA                        //存放数据集格式 (可放在别的文件夹)
│   │   ├── train                   //训练集图片文件夹
│   │   │   ├── images
│   │   │   │   ├── 00001__800__0___1300.png
│   │   │   │   └── ...
│   │   │   └── labelTxt
│   │   │       ├── 00001__800__0___1300.txt
│   │   │       └── ...
│   │   ├── valid                   //验证集图片文件夹
│   │   │   ├── images
│   │   │   │   ├── 00001__800__0___1300.png
│   │   │   │   └── ...
│   │   │   └── labelTxt
│   │   │       ├── 00001__800__0___1300.txt
│   │   │       └── ...
│   │   └── test                    //测试集图片文件夹
│   │       ├── images
│   │       │   ├── 00001__800__0___1300.png
│   │       │   └── ...
│   │       └── labelTxt
│   │           ├── 00001__800__0___1300.txt
│   │           └── ...
│   └── ...
├── Models                      //构建不同的模型
│   ├── DBNet                   //每个模型都创建一个文件夹
│   │   ├── DBNet.py            //模型文件
│   │   ├── module_backbone.py  //骨干网络等模块
│   │   ├── module_det_head.py  //检测头等模块
│   │   ├── module_loss.py      //构建损失函数
│   │   └── ...   
│   ├── pretrained              //预训练权重文件夹.对网络初始化时使用
│   ├── reference               //仅仅参考别的模型,不参与上面的模型构建
│   └── ...      
├── Tools                       //存放较大的,独立的工具类和文件
│   ├── main.py                 //训练,验证,测试入口
│   └── ...     
├── Utils                       //存放规模较小,相对独立的轻量级工具和类文件
│   ├── IO_Utils                //输入输出工具类
│   ├── OOD_Postprocessing      //后处理工具类
│   ├── Trainer                 //(重要)模型训练工具类
│   │   ├── Trainer_base        //保存不同类型的权重文件
│   │   │   ├── base_trainer.py //基类训练流程文件,各个模型不同的训练会继承这个文件
│   │   │   └── metrics.py      //评价指标文件
│   │   ├── Trainer_OOD         //旋转目标检测类
│   │   └   └── trainer.py      //继承上面的基类,进行针对性的改进
│   └   ...                     //不同的实验输出文件夹   
├── Output                      //输出文件夹
│   ├── SavaDir                 //不同模型保存文件夹，如果重名了进行删除
│   │   ├── checkpoints         //保存不同类型的权重文件
│   │   │   ├── best_99.pkl     //权重文件,最好的，必须存在
│   │   │   ├── latest_99.pkl   //权重文件，最后的，必须存在
│   │   │   └── epoch_99.pkl    //权重文件，制定间隔的，可选
│   │   └── log                 
│   │       ├── event           //tensorboard文件(训练日志记录)
│   │       ├── config.py       //本次训练所用的参数信息，可以直接替换对应的训练配置下面
│   │       ├── Predict.txt     //训练结束后对数据集的评测文件
│   │       └── train_log.txt   //训练日志信息
│   └   ...                     //不同的实验输出文件夹
│── Readme_imgs                 //存放README.md中进行展示的图片
├── README.md                   //介绍
└── requirements.txt            //包依赖

```






在 Ubuntu 20.04、CUDA 11.3 上使用 python3.9 进行测试 
（tensorboard、tqdm、opencv-python、opencv-python-headless、scikit-image)

[//]: 

```bash

conda create -n HTD-SOTA python=3.9

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt -i https://pypi.doubanio.com/simple
```


下载[DOTA](https://captain-whu.github.io/DOTA/dataset.html)数据并进行转换,目录结构应如下所示
,详细介绍请[跳转](Datasets/README.md)
：

|            | train  | validation | test  |
|:----------:|:------:|:----------:|:-----:|
| ***DOTA*** |  1411  |    458     |  937  |
|  DOTA-RR   | 15749  |    5297    | 10833 |
|  DOTA-MS   | 103421 |   35462    | 71888 |
```
./DOTA
├── train
│   └── images
│   └── labelTxt
├── val
│   └── images
│   └── labelTxt
├── test
│   └── images
└   └── labelTxt
```


1. 更改配置文件相关参数,配置文件介绍,请[跳转](Configs/README.md)
2. 尝试训练代码:如果设备上有多个 GPU，请设置 'CUDA_VISIBLE_DEVICES=0' 以使用它们。如果不使用将调用所有显卡进行分布式计算,例如

```
CUDA_VISIBLE_DEVICES=1 python /home/sonel/code/Sonel_code/HTD-SOTA/Tools/main.py --config_file /home/sonel/code/Sonel_code/HTD-SOTA/Configs/SODA_r50_DBCNet_800_6x.yaml
CUDA_VISIBLE_DEVICES=0 python /home/sonel/code/Sonel_code/HTD-SOTA/Tools/main.py --config_file /home/sonel/code/Sonel_code/HTD-SOTA/Configs/HRSC_DBCNet_6x.yaml
```
3. 输出模型和日志文件将存储在 'Output' 中,可以使用tensorboard进行可视化中间过程。
```
tensorboard --logdir ./log
```


1. 修改相关配置文件为eval或test,代码运行参照训练流程:




```bash

ssh ubuntu@xxx.xxx.cn  
```
```bash
scp -r ubuntu@xxx.xxx.cn:~/work.pdf /home/sonel/下载/  
```


```bash
nvidia-smi -l 
```
```bash
top 
```
```bash
watch sensors  
```



mmrotate框架：[mmrotate](https://mmrotate.readthedocs.io/zh-cn/1.x/get_started.html)  
