# 学习率扫描器 (Learning Rate Finder)

学习率扫描器是一个帮助找到最优学习率的工具，基于 Leslie Smith 的论文 ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186)。

## 📚 原理

学习率扫描器通过以下步骤工作：

1. **指数增长**: 从非常小的学习率（如 1e-7）开始，逐步指数增加到较大的学习率（如 1.0）
2. **短期训练**: 在每个学习率下训练几个批次，记录损失值
3. **曲线分析**: 绘制学习率-损失曲线，找到损失下降最快的区域
4. **推荐学习率**: 基于曲线特征给出学习率建议

## 🚀 快速开始

### 1. 运行学习率扫描

```bash
# 使用随机初始化的模型
cargo run --release --bin banqi-lr-finder

# 或者加载已有模型
cargo run --release --bin banqi-lr-finder banqi_model_latest.ot
```

**注意**: 需要先运行训练程序生成训练样本数据库 (`training_samples.db`)。

### 2. 可视化结果

使用 Python 脚本绘制学习率-损失曲线：

```bash
# 安装依赖（首次运行）
pip install pandas matplotlib numpy

# 绘制曲线
python plot_lr_finder.py
```

这将生成：
- `lr_finder_plot.png` - 包含 4 个子图的综合分析图
- 命令行输出的学习率建议

### 3. 查看数据

学习率扫描结果保存在 `lr_finder_results.csv`：

```csv
learning_rate,loss,policy_loss,value_loss
1.00000000e-07,2.345678,1.234567,1.111111
2.15443469e-07,2.340123,1.230456,1.109667
...
```

## 📊 图表解读

可视化工具生成 4 个子图：

### 1. Total Loss vs Learning Rate
- 展示总损失随学习率的变化
- **红色虚线**: 最小损失点
- **绿色虚线**: 损失下降最快的点

### 2. Policy & Value Loss Components
- 分别显示策略损失和价值损失
- 帮助理解两个损失分量的行为差异

### 3. Loss Gradient
- 显示损失对 log(学习率) 的梯度
- 最负的梯度点即为损失下降最快的位置

### 4. Recommended Learning Rate Range
- **黄色阴影区域**: 推荐的学习率范围
- **橙色实线**: 建议的初始学习率
- **绿色虚线**: 建议的最小学习率
- **红色虚线**: 建议的最大学习率

## 🎯 如何选择学习率

### 理想的学习率-损失曲线

```
Loss
  |
  |     \
  |      \___
  |          \___
  |              \___________/
  |__________________________|___
                              Learning Rate (log scale)
     区域1   区域2        区域3
```

- **区域 1**: 损失几乎不变（学习率太小）
- **区域 2**: 损失快速下降（**最佳区域**）
- **区域 3**: 损失发散（学习率过大）

### 推荐策略

1. **单一学习率训练**
   - 使用建议的初始学习率
   - 示例: `let learning_rate = 5e-5;`

2. **学习率衰减**
   - 从初始学习率开始，逐步降低
   - 示例: 每 N 轮降低为原来的 0.5-0.9 倍

3. **循环学习率 (Cyclic LR)**
   - 在最小和最大学习率之间周期性变化
   - 示例: 使用三角形或余弦退火调度

4. **One-Cycle 策略**
   - 从最小学习率逐渐增加到最大学习率，然后快速降低
   - 适合训练周期较短的场景

## ⚙️ 配置选项

可以在 `run_lr_finder.rs` 中修改 `LRFinderConfig`：

```rust
let config = LRFinderConfig {
    start_lr: 1e-7,              // 起始学习率
    end_lr: 1.0,                 // 结束学习率
    num_steps: 100,              // 扫描步数（采样点）
    num_batches_per_step: 2,     // 每个学习率训练的批次数
    batch_size: 64,              // 批量大小
    smooth_window: 5,            // 平滑窗口大小
    divergence_threshold: 4.0,   // 发散阈值（4倍最小损失）
};
```

### 参数说明

- **start_lr**: 起始学习率，通常设为 1e-7 到 1e-8
- **end_lr**: 结束学习率，通常设为 1.0 到 10.0
- **num_steps**: 学习率采样点数量，越多曲线越平滑，但耗时越长
- **num_batches_per_step**: 每个学习率训练的批次数，通常 1-3 个
- **batch_size**: 批量大小，与训练时保持一致
- **smooth_window**: 移动平均窗口，减少曲线噪声
- **divergence_threshold**: 如果损失超过最小损失的此倍数，提前停止

## 🔧 应用到训练代码

在 `parallel_train.rs` 中应用新学习率：

```rust
// 修改前
let learning_rate = 1e-4;

// 修改后（使用扫描器建议的学习率）
let learning_rate = 5e-5;  // 根据扫描结果调整
let mut opt = nn::Adam::default().build(&vs, learning_rate)?;
```

## 📝 注意事项

### 1. 何时运行学习率扫描

- **初次训练**: 在开始大规模训练之前运行
- **架构变更**: 修改模型结构后重新扫描
- **数据变化**: 训练数据分布变化时
- **训练不稳定**: 遇到损失震荡或发散时

### 2. 样本数量

- 扫描器默认使用最新的 10,000 个样本
- 样本数量影响扫描结果的准确性
- 如果样本不足，可能需要先运行几轮训练

### 3. 模型加载

- 可以从头开始扫描（随机初始化）
- 也可以加载已训练模型继续扫描
- 建议在训练初期使用随机初始化扫描

### 4. 结果解读

- **曲线平滑**: 说明学习率范围合理
- **曲线抖动**: 可能需要增加 `smooth_window`
- **无明显下降**: 可能模型过拟合或样本质量低
- **快速发散**: 正常现象，说明找到了学习率上限

## 🔬 高级用法

### 自定义样本筛选

修改 `run_lr_finder.rs` 以使用特定样本：

```rust
// 只使用结局前 N 步的样本
let samples_to_use: Vec<_> = samples.into_iter()
    .filter(|(obs, _, _, _)| {
        // 添加自定义筛选逻辑
        true
    })
    .collect();
```

### 多次扫描

在不同阶段运行多次扫描，比较结果：

```bash
# 训练前
cargo run --release --bin banqi-lr-finder
mv lr_finder_results.csv lr_finder_initial.csv

# 训练 5 轮后
cargo run --release --bin banqi-lr-finder banqi_model_5.ot
mv lr_finder_results.csv lr_finder_epoch5.csv

# 训练 20 轮后
cargo run --release --bin banqi-lr-finder banqi_model_20.ot
mv lr_finder_results.csv lr_finder_epoch20.csv
```

### 与其他优化器配合

学习率扫描器使用 Adam 优化器，如果训练时使用其他优化器（如 SGD），建议：

1. 在扫描器中也改用相同优化器
2. 或者根据优化器特性调整建议学习率

## 📚 参考资料

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
- [A disciplined approach to neural network hyper-parameters](https://arxiv.org/abs/1803.09820)
- [fastai Learning Rate Finder](https://docs.fast.ai/callback.schedule.html#LRFinder)

## 🐛 故障排除

### 问题 1: "数据库中没有训练样本"

**解决方案**: 先运行训练程序生成样本

```bash
cargo run --release --bin banqi-train  # 或 banqi-parallel-train
```

### 问题 2: "损失曲线没有明显变化"

**可能原因**:
- 样本质量低
- 模型已经过拟合
- 学习率范围设置不当

**解决方案**:
- 增加训练样本数量
- 扩大学习率扫描范围（如 `start_lr: 1e-8, end_lr: 10.0`）
- 检查样本分布（使用 `banqi-verify-samples`）

### 问题 3: "扫描过程很慢"

**解决方案**:
- 减少 `num_steps`（如降到 50）
- 减少 `num_batches_per_step`（设为 1）
- 限制样本数量（默认已限制为 10,000）

### 问题 4: "曲线很平滑但找不到最优学习率"

**解决方案**:
- 增加 `num_steps` 以获得更细粒度的曲线
- 调整 `smooth_window` 大小
- 手动检查 CSV 文件，找到梯度最负的区域

## 📞 获取帮助

如果遇到问题或有改进建议，请：

1. 检查本文档的故障排除部分
2. 查看 `lr_finder_results.csv` 中的原始数据
3. 使用 `python plot_lr_finder.py` 可视化结果
4. 参考相关论文和文档

---

**祝训练顺利！** 🚀
