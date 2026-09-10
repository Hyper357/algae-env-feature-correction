# Data interface

原始实验数据不提交到本仓库。

## `stations.csv`

至少包含：

```text
station_id,chlorophyll,salinity,pH,temperature,DO,ORP,turbidity
```

字段可以缺失，但运行某一环境变量时该列必须存在。

## `features.csv`

至少包含：

```text
station_id,f_001,f_002,...,f_n
```

`f_*` 为从 F97 Pro EEM 提取的有限通道荧光特征、归一化特征或机理比值特征。

## 数据原则

- 每个站点只出现一次；
- 不提交可识别未发表实验原始文件；
- 任何预处理参数必须由训练折拟合；
- 若叶绿素作为 target，则不能同时作为 environment input。
