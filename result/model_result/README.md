该文件包含第五次 5 叉验证的模型参数，可以直接导入模型并进行测试

# 目录

1 **fastGTN_{i}.pth**: 在第 i 折中保留的模型

2 **pred_label.csv**: 药物-疾病关联的预测标签，第一列是行索引，第二列是 0/1 标签，第一行是列索引，保存过程中标题和索引一起保存。0 表示没有关联，1 表示有关联。

3 **true_label.csv**: 药物-疾病关联的真实标签，第一列是行索引，第二列是 0/1 标签，第一行是列索引，保存过程中标题和索引一起保存。0 表示没有关联，1 表示有关联。