

# Publication
__Rui Sun__, De-Min Liang, Pu-Feng Du*. DDA-GTN: large-scale drug repurposing on drug-gene-disease heterogenous association networks using graph transformers. _IEEE Journal of Biomedical and Health Informatics_ (2024). (Revised)

# 基于图Transformer的药物-疾病关联预测方法

在这项工作中，我们首先提出了一个基准数据集，其中包括构成三层异构网络的药物、基因和疾病三个实体，并引入图 Transformer网络（Graph Transformers Networks）来学习异构网络中药物-疾病的低维特征表示，关联分布编码器预测药物-疾病关联。我们将这种方法命名为 DDA-GTN。
 [英文版GitHub链接](https://github.com/SunRuir/DDA-GTN).

# 1.实验平台和工具包

## 1.1 实验平台

- ubuntu 18.04

- RTX 3090(24GB)

## 1.2 工具包

| 名称     | 版本号                                |
| --------- | ----------------------------------- |
| CUDA     | 11.3                     |
| Python     | 3.9.13                     |
| torch     | 1.11.0                     |
| torch_geometric     | 2.1.0.post1                     |
| torch-scatter     | 1.6.0                     |
| torch-sparse     | 0.6.15                     |
| torch-cluster     | 1.6.0                     |
| pandas     | 1.4.4                     |
| scikit-learn     | 1.1.2                     |
| matplotlib     | 3.6.0                     |

# 2. 目录结构

## 2.1 src

> 该文件夹存储代码文件。

## 2.2 result

> 该文件夹包含五次五折交叉验证的运行记录、第五次交叉验证的模型参数和预测值、计算五次交叉验证平均结果和标准偏差的代码，以及绘制 AUC 和 ROC 曲线的代码。

## 2.3 compare

> 本文件包含 LAGCN、LHGCE 和 REDDA 的运行代码，这些代码来自相应出版物公开的 github 库。

# 3. 工作流程

## 3.1安装环境

```
pip install -r request.txt
```

## 3.2 下载基准数据集M

基准数据集下载链接为 [Zenodo](https://zenodo.org/records/10827427).

将数据集文件夹放在项目的根目录。项目结构如下：

```
/DDA-GTN
  |-Data
  |-src
  |-result
  |-compare
```

### 数据目录

数据在 zenodo 中，其中包含 Mdataset 对 DDA-GTN 进行 5 次交叉验证的输入数据。

zenodo 中的 Data 文件夹应与 src 文件夹并列存放。

| 文件名      | 描述                                |
| --------- | ----------------------------------- |
| C_D.csv     | 药物-疾病关联 <br>  CTD IDs -- MeSH IDs                     |
| C_G.csv     | 药物-基因关联 <br>  CTD IDs -- Gene Symbol                     |
| G_D.csv     | 基因-疾病关联 <br>  Gene Symbol -- MeSH IDs -- InferenceScore                    |
| disease_feature.csv     | 疾病特征矩阵 2447*881                      |
| drug_feature.csv     | 药物特征矩阵 5973*881                    |
| gene_feature.csv     | 基因特征矩阵 12582*881                 |
| A_CD.csv     | 药物-疾病关联矩阵 5973*2447 ，5973 代表药物类别数，2447 代表疾病类别数    |
| node_list.csv     | 它按照药物（CTD IDs）、基因（Gene Symbol）和疾病（MeSH IDs）的顺序包含了异构网络中的所有节点，节点对应的位置就是稀疏矩阵中的索引。                   |
| NegativeSample.csv     | 从药物-疾病关联矩阵中随机抽取与阳性样本相同数量的阴性样本 <br> 药物索引 -- 疾病索引 |

## 3.3 负采样

> python src/negativesample.ipynb

## 3.4 划分数据集

> python src/split.py seed **

> 要运行这段代码，需要设置随机种子和保存路径，路径需要在运行前手动创建

## 3.5 疾病特征生成

> python src/feature_ge_Cycle.py

> feature0{i} 是一个文件夹名称，可在运行代码时设置，也可更改为其他名称。

> 设置要读入的五个文件夹路径，读入的路径必须与保存由 src/split.py 生成的文件的路径相同

## 3.6 交叉验证及重定位预测

数据文件夹中提供了 Mdataset 上述三个步骤的数据，因此可以直接执行该代码。

### 3.6.1 交叉验证

> python src/MdataNW_5cross.py

> 这将分别在 saving_path/models 和 saving_path/log.txt 中保存模型和记录。

> 需要设置读入数据集和疾病特征的路径。

> 读取 5 个交叉验证数据的路径必须与 split.py 生成的文件路径相同

> 读取疾病特征数据的路径必须与 feature_ge_Cycle.py 生成的文件路径相同

#### 参数设置

- epoch: 默认值=100. The number of training epoch.

- lr: 默认值=0.005. The initial learning rate.

- weight_decay: 默认值=5e-4. The weight decay for this training.

- node_dim: 默认值=128. The dim for node feature matrix.

### 3.6.2 重定位预测

所有结果都保存在 Zenodo 中，下载链接为 [Zenodo](https://zenodo.org/records/10827427).

> python src/casestudy_Mdata.py

> 分别在 saving_path/models 和 saving_path/log.txt 中保存模型和日志。


