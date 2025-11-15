


<div align=center><img src="img_demo/00001__800__3250___0.jpg"  width=30%>
<img src="img_demo/00003__800__4000___0.jpg"  width=30%>
<img src="img_demo/00004__800__4000___650.jpg"  width=30%>
</div>
<div align=center>DOTA数据集Demo</div>


```
│   ├── DOTA                     //存放数据集格式 (可放在别的文件夹)
│   │   ├── train               //训练集图片文件夹
│   │   │   ├── images
│   │   │   │   ├── 00001__800__0___1300.png
│   │   │   │   └── ...
│   │   │   └── labelTxt
│   │   │       ├── 00001__800__0___1300.txt
│   │   │       └── ...
│   │   ├── valid               //验证集图片文件夹
│   │   │   ├── images
│   │   │   │   ├── 00001__800__0___1300.png
│   │   │   │   └── ...
│   │   │   └── labelTxt
│   │   │       ├── 00001__800__0___1300.txt
│   │   │       └── ...
│   │   └── test                //测试集图片文件夹
│   │       ├── images
│   │       │   ├── 00001__800__0___1300.png
│   │       │   └── ...
│   │       └── labelTxt
│   │           ├── 00001__800__0___1300.txt
│   │           └── ...
```

- x1-y4:表示旋转目标的坐标位置。 
- xclass:表示旋转目标的种类。
- flag:表示对目标的保留参数。
   - 0:代表正式需要识别的类型，如：飞机，小汽车等。   
   - -1:代表过滤该目标(忽略模式)。  
   - 备注：这里可以通过置不同的参数进行不同的数据集自定操作。
   - 如是否创建阈值图，进行过滤目标(掩膜模式)。

```
例子：
x1,y1,x2,y2,x3,y3,x4,y4,class_idx,flag
如：
937 913 921 912 923 874 940 875 small-vehicle 0
638 959 638 935 694 939 693 962 large-vehicle 0
545 494 548 518 489 519 488 493 large-vehicle 0
536 468 535 489 477 486 478 464 large-vehicle 0
```



Transform_OOD  



```
  self.samples: list, 多个个图像有多少个元素
        - 'filename': str, 图像文件名字，不包含前置路径，包含后置格式
          'ann'：dict, 原始标注图像数据
              'bbox': np, (n,5) n代表该图像有多少个目标，5代表中心点+长宽+旋转角度
              'labels': np, (n,) n的每个元素代表目标的种类下标
              'polygons': np, (n,4,2) 4和2分别代表着四个点，每个点的xy值
              'polygons_ignore'： np, 同上，忽略点信息
              'labels_ignore'： np, 同上，忽略标签类别信息
              'bbox_ignore'： np, 同上，忽略目标框信息
              
              iaa_ploygons:使用iaa方法对polygons里面的点进行变换。
          ’img‘: np, (h,w,c) 宽高通道
          
              
  self.ori_img_ids：list，将self.sample的filename信息，去掉后缀得到的标号id
  self.mean, self.std: float, 整个数据集的均值和方差数据。如果在字段中则不需要进行计算。
   
  self.dataset_class: list, 种类字符串列表
  self.cls_ids: dict, 通过下标查找对应的种类字符串
  self.id_clss: dict, 通过字符串查找对应的下标
  self.angle_version: str, 坐标转换方式
  
  self.params: dict, 通过配置文件传过来的配置信息，仅包含dataset字段
  
```


想到的额外的点：
1. 类的 __repr__() 方法定义了实例化对象的输出信息
2. 可以在数据集加载的时候增加一个开关，可以控制图像数据是在Init的时候被加载还是在getitem的时候被加载。
3. cv2.imwrite("img_gt.jpeg",np.transpose(np.array(batch['img'].cpu().detach())[0,...]*255,(1,2,0)))```
4. 可以通过num_worker进行并行化加载数据.一般设置为可用GPU的两倍
5. 这里可以在数据加载的时候进行创建阈值图等操作,但是不应该在前向传播的时候进行操作,最直观的例子就是不能进行并行处理.  
6. 目前处理4点。以后数据集处理annotation的时候可以考虑下多点输入的情况。前面处理标签数据，后面-2位置置为类别，-1位置置为flag。




数据集格式
```
937 913 921 912 923 874 940 875 small-vehicle 0
```
最后两个数据分别代表着数据集标签和flag信息  
如果为-1代表忽略。对之加上相应的掩膜信息。  
后面增加的类型可以再进行讨论  

dataset = eval(data['train'].pop("type"))(**data['train'])  


SODA数据集处理
1.合并zip图像  tar cvfz out.zip Images.zip Images.z01 Images.z02
2.去除多余标签数据.得到wo文件