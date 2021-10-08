## 训练
在 config.json中配置好文件夹和参数

使用命令
`python train.py train_name`
以开始训练

train_name 会决定保存checkpoint时的文件名

## 参考文献
https://arxiv.org/abs/1704.04368

## 目前问题
生成质量较差 > 增加数据量和质量
生成长摘要时质量较差 > 调整scheduled sampling策略