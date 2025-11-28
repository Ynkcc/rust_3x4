#!/usr/bin/env python3
"""
学习率扫描结果可视化工具

使用方法:
    python plot_lr_finder.py [lr_finder_results.csv]

功能:
1. 读取 lr_finder_results.csv 文件
2. 绘制学习率-损失曲线
3. 标记关键点（最小损失点、最陡下降点）
4. 保存为图片文件

依赖:
    pip install pandas matplotlib numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_lr_finder(csv_path='lr_finder_results.csv', output_path='lr_finder_plot.png'):
    """
    绘制学习率扫描结果
    
    Args:
        csv_path: CSV 文件路径
        output_path: 输出图片路径
    """
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        print("提示: 请先运行 'cargo run --bin banqi-lr-finder' 生成结果文件")
        return
    
    # 读取数据
    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("❌ CSV 文件为空")
        return
    
    print(f"✓ 成功读取 {len(df)} 个数据点")
    
    # 提取数据
    lr = df['learning_rate'].values
    loss = df['loss'].values
    policy_loss = df['policy_loss'].values
    value_loss = df['value_loss'].values
    
    # 找到关键点
    min_loss_idx = np.argmin(loss)
    min_loss_lr = lr[min_loss_idx]
    min_loss_val = loss[min_loss_idx]
    
    # 计算梯度 (d(loss)/d(log_lr))
    log_lr = np.log(lr)
    gradients = np.gradient(loss, log_lr)
    max_gradient_idx = np.argmin(gradients)  # 最负的梯度
    steepest_lr = lr[max_gradient_idx]
    steepest_loss = loss[max_gradient_idx]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('学习率扫描结果 (Learning Rate Finder)', fontsize=16, fontweight='bold')
    
    # 子图 1: 总损失曲线
    ax1 = axes[0, 0]
    ax1.plot(lr, loss, 'b-', linewidth=2, label='Total Loss')
    ax1.axvline(min_loss_lr, color='r', linestyle='--', alpha=0.7, label=f'Min Loss LR: {min_loss_lr:.2e}')
    ax1.axvline(steepest_lr, color='g', linestyle='--', alpha=0.7, label=f'Steepest LR: {steepest_lr:.2e}')
    ax1.plot(min_loss_lr, min_loss_val, 'ro', markersize=10, label=f'Min Loss: {min_loss_val:.4f}')
    ax1.plot(steepest_lr, steepest_loss, 'go', markersize=10, label=f'Steepest: {steepest_loss:.4f}')
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Total Loss vs Learning Rate', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 子图 2: 策略损失和价值损失
    ax2 = axes[0, 1]
    ax2.plot(lr, policy_loss, 'b-', linewidth=2, label='Policy Loss', alpha=0.7)
    ax2.plot(lr, value_loss, 'r-', linewidth=2, label='Value Loss', alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel('Learning Rate', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Policy & Value Loss Components', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 子图 3: 损失梯度
    ax3 = axes[1, 0]
    ax3.plot(lr[1:], gradients[1:], 'purple', linewidth=2)
    ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(steepest_lr, color='g', linestyle='--', alpha=0.7, label=f'Steepest: {steepest_lr:.2e}')
    ax3.set_xscale('log')
    ax3.set_xlabel('Learning Rate', fontsize=12)
    ax3.set_ylabel('Gradient (d(loss)/d(log_lr))', fontsize=12)
    ax3.set_title('Loss Gradient', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # 子图 4: 建议区域（放大视图）
    ax4 = axes[1, 1]
    
    # 计算建议的学习率范围
    suggested_min_lr = steepest_lr
    suggested_max_lr = min_loss_lr / 3.0
    suggested_initial_lr = np.sqrt(suggested_min_lr * suggested_max_lr)
    
    # 找到这个范围内的数据
    mask = (lr >= suggested_min_lr / 2) & (lr <= min_loss_lr * 2)
    lr_zoom = lr[mask]
    loss_zoom = loss[mask]
    
    ax4.plot(lr_zoom, loss_zoom, 'b-', linewidth=2)
    ax4.axvline(suggested_initial_lr, color='orange', linestyle='-', linewidth=2, 
                label=f'Suggested Initial LR: {suggested_initial_lr:.2e}')
    ax4.axvline(suggested_min_lr, color='g', linestyle='--', alpha=0.7, 
                label=f'Min LR: {suggested_min_lr:.2e}')
    ax4.axvline(suggested_max_lr, color='r', linestyle='--', alpha=0.7, 
                label=f'Max LR: {suggested_max_lr:.2e}')
    
    # 添加阴影区域表示建议范围
    ax4.axvspan(suggested_min_lr, suggested_max_lr, alpha=0.2, color='yellow', 
                label='Recommended Range')
    
    ax4.set_xscale('log')
    ax4.set_xlabel('Learning Rate', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Recommended Learning Rate Range', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    # 打印总结
    print("\n" + "="*50)
    print("学习率建议总结")
    print("="*50)
    print(f"最小损失点:")
    print(f"  学习率: {min_loss_lr:.2e}")
    print(f"  损失: {min_loss_val:.4f}")
    print(f"\n损失下降最快点:")
    print(f"  学习率: {steepest_lr:.2e}")
    print(f"  损失: {steepest_loss:.4f}")
    print(f"\n推荐学习率:")
    print(f"  初始学习率: {suggested_initial_lr:.2e}")
    print(f"  最小学习率: {suggested_min_lr:.2e}")
    print(f"  最大学习率: {suggested_max_lr:.2e}")
    print("="*50)

def main():
    """主函数"""
    # 获取命令行参数
    csv_path = 'lr_finder_results.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    # 绘制图表
    plot_lr_finder(csv_path)

if __name__ == '__main__':
    main()
