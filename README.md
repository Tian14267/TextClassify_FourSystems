# TextClassify_FourSystems
## 说明
本文主要是实现`NLP`文本分类任务，目前该系统为四系统融合模型，其中包括基于字向量Char的CNN模型和RCNN模型以及基于词向量word2vec的CNN模型和RCNN模型。目前的融合则采用权值平均法进行融合。

## 环境
    python3
    tensorflow

## 数据
所使用的数据为网上的cnews数据，共10个类别，分为训练集，验证集和测试集。 类别如下：

体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐

训练集每个类别5千数据。共5万。验证集每个类别500，测试集每个类别1000。 训练集、验证集和测试集的文件如下：
    cnews.train.txt: 训练集(50000条)
    cnews.val.txt: 验证集(5000条)
    cnews.test.txt: 测试集(10000条)

## Word2vec
本文这里公开了自己使用的`word2vec`模型，也可以自己根据自己的数据集制作词向量模型。
链接：https://pan.baidu.com/s/1dYRIPVH1N0y0gNOCw8rAuQ  提取码：g889

## 模型训练
训练情况如下图：
![image](https://github.com/Tian14267/TextClassify_FourSystems/blob/master/images/666.png)

## 测试
下面为模型测试效果图：

![image](https://github.com/Tian14267/TextClassify_FourSystems/blob/master/images/777.png)
