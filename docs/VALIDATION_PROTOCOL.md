# 环境状态驱动的荧光特征校正：验证协议 v0.1

## 1. 研究问题

当前需要回答的不是“环境参数能不能预测藻”，而是：

> 在真实海水环境中，盐度、pH、水温等状态变量是否导致有限通道荧光特征发生系统性漂移；若存在，能否通过显式的环境状态校正提高荧光测量对独立生物量指标的跨站点泛化能力？

这是测量补偿问题，不是生态相关性分类问题。

## 2. 数据分层

### 2.1 外采 F97 Pro 数据

用途：环境补偿主验证集。

要求：

- 每个站点保留唯一 `station_id`；
- EEM 预处理流程必须固定；
- 从 EEM 中提取与后续硬件可实现波段相一致的有限通道特征；
- 同步环境参数至少包括：盐度、pH、水温、DO、浊度、ORP；
- 叶绿素可作为独立验证终点，但此时不能作为补偿输入。

### 2.2 F-4700 受控藻数据

用途：说明有限通道荧光特征最终可用于光学藻群识别，并提供既有算法基线。

限制：

- 无同步盐度/pH梯度，不能证明环境补偿效果；
- 与 F97 Pro 不做强制跨仪器数值拼接；
- 只作为“校正后特征的下游应用场景”证据。

## 3. 特征层级

按以下顺序验证，禁止一开始堆高维模型：

### A. 原始有限通道强度

例如：

```text
I(Ex, Em)
```

### B. 幅值归一化特征

用于削弱总信号强度差异。

### C. 机理比值特征

例如：

```text
log((I_580 + eps) / (I_680 + eps))
log((I_730 + eps) / (I_680 + eps))
```

环境补偿优先作用于 B/C 层，而不是直接修改整张 EEM。

## 4. 环境变量优先级

第一轮逐变量验证：

1. salinity
2. pH
3. temperature
4. DO
5. ORP
6. turbidity

原因：小样本下必须先识别真正有增量价值的环境状态变量，避免参数堆叠。

## 5. 三组模型

每个环境变量 `z` 必须同时比较三组。

### M0：荧光基线

```text
y = h(x)
```

### M1：直接拼接诊断模型

```text
y = h([x, z])
```

用途仅为筛查“z 是否携带增量信息”，**不是候选专利核心实现**。

### M2：显式环境校正模型

在训练集上，对每个特征 `x_j` 建立：

```text
g_j(z)
```

以训练集环境中位数作为参考状态 `z0`：

```text
x_corr[j] = x[j] - (g_j(z) - g_j(z0))
```

随后：

```text
y = h(x_corr)
```

如果 M2 能稳定优于 M0，并接近或超过 M1，说明“显式补偿”有数据依据。

## 6. 外层验证

当前站点数较少，主验证采用 Leave-One-Station-Out：

```text
for each station s:
    train = all stations except s
    test  = station s

    fit preprocessing on train
    fit g_j on train
    correct train/test using train-fitted parameters
    fit downstream model on corrected train
    predict test
```

禁止：

- 在全数据上先标准化；
- 在全数据上先拟合环境补偿函数；
- 根据测试站点表现选择环境变量；
- 先看结果再决定哪些站点属于异常值。

## 7. 下游验证终点

### 第一阶段主终点

独立叶绿素测量值。

评价：

- MAE
- RMSE
- R²
- Spearman rho

注意：此阶段验证的是**光学测量补偿价值**，不是藻群识别准确率。

### 第二阶段终点

需要未来有同步藻群真值后才能正式验证：

- 光学群组 exact-set accuracy
- macro-F1
- per-group recall
- weak-component recall

## 8. 负对照

### 8.1 环境变量置换

在保持 EEM 与叶绿素配对不变的条件下，随机打乱环境变量 `z`，重复至少 1000 次。

统计：

```text
Delta_MAE = MAE_baseline - MAE_corrected
p_perm = fraction(Delta_MAE_perm >= Delta_MAE_real)
```

### 8.2 环境变量单独模型

```text
y = h(z)
```

若 `z` 单独就能预测目标，而 M2 没有额外价值，则更可能是生态共变，而不是荧光补偿。

### 8.3 极端站点敏感性

重复：

- 全部站点；
- 去掉目标最高站点；
- 去掉盐度最低/最高站点；
- 去掉影响度最高的 1–2 个站点。

要求改善方向稳定。

## 9. 推荐最小补偿模型

小样本阶段优先线性/低自由度模型：

```text
g_j(z) = beta0_j + beta1_j * z
```

若线性模型通过后，再比较：

- ridge
- quadratic polynomial
- monotonic spline

不建议直接上深度网络或高阶树模型。

## 10. Go / No-Go

进入专利实施例准备的最低条件建议：

```text
1) LOSO MAE relative improvement >= 15%
2) R² improves in same direction
3) permutation p < 0.05
4) sensitivity analyses retain same sign
5) M2 benefit cannot be explained by z-only model
6) at least one physically interpretable fluorescence feature shows stable correction relation
```

若只满足 M1，不满足 M2：结论为“环境参数有预测价值”，但**不立环境补偿专利**。

若 M2 通过：进入 P02，冻结校正公式、环境变量、特征集合与实施例。
