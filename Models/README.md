

查看模型是否在CUDA上
self.models.backbone.bn1.weight.device
获得模型名称 :  model.name
获得参数量以及模型 from thop import profile

在处理模型的时候.backbone需要单独的文件夹.neck需要单独的文件夹.DBNet需要单独的
模型应该是这样进行分类的.

```
├── Model                       //公有的模型，在此基础上进行更改,不加入到最后的模型中
│   ├── __init__.py             //文件夹里面的内容暴露在外面
│   ├── backbone                //不同模型
│   ├── neck                    //不同模型
│   ├── DBMFormer               //不同模型
│   └── ...                      
```


获得

构建模型流程.
1.构建模型需要的参数,直接写入,尽量不进行config转换.
2.构建完模型之后,在Models.init部分进行模型加入
3.构建配置文件,对模型进行配置文件构建方法
4.对模型进行训练,小规模数据集测试,之后转移到大规模下.
