import json
import pandas as pd
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载数据 (使用 pandas 更快捷)
file_path = "experiment_results_v3.jsonl"
try:
    df = pd.read_json(file_path, lines=True)
except Exception as e:
    print(f"加载失败，请检查文件格式: {e}")
    # 备用方案
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

full_rank = 1000

# 设置全局绘图风格
sns.set_theme(style="whitegrid", palette="muted")
# 解决某些后端显示问题，确保导出时不触发 tostring_rgb 错误
plt.rcParams['figure.facecolor'] = 'white'

# 创建画布：1行3列
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- 图 1: Self Rank 的累积分布函数 (CDF) ---
# 这是衡量 EIF 效果最核心的图：曲线越靠左上角，说明“自相关”识别越准
sns.ecdfplot(data=df, x='self_rank', ax=axes[0], lw=2, color='blue')
axes[0].set_title('Cumulative Distribution of Self Rank\n(Steeper = Better)', fontsize=14)
axes[0].set_xlabel('Rank')
axes[0].set_ylabel('Proportion of Samples')

# --- 图 2: Score 的分布 (带 Log 轴可选) ---
sns.histplot(df['score'], kde=True, ax=axes[1], color='salmon')
# axes[1].set_yscale('log')
axes[1].set_title('Distribution of Influence Scores', fontsize=14)
axes[1].set_xlabel('Score')


hit_top_1_percent  = (df['self_rank'] >= full_rank * 0.99).mean() * 100
hit_top_5_percent  = (df['self_rank'] >= full_rank * 0.95).mean() * 100
hit_top_10_percent = (df['self_rank'] >= full_rank * 0.9).mean() * 100
labels = ['Top 1%', 'Top 5%', 'Top 10%']
values = [hit_top_1_percent, hit_top_5_percent, hit_top_10_percent]

sns.barplot(x=labels, y=values, ax=axes[2], palette="viridis")
axes[2].set_title('Model Success Rate at Different Thresholds', fontsize=14)
axes[2].set_ylabel('Success Rate (%)')
axes[2].set_ylim(0, 100) # 纵坐标固定 0-100
for i, v in enumerate(values):
    axes[2].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# --- 统计报告 ---
print("\n" + "="*30)
print("   EXPERIMENTAL SUMMARY")
print("="*30)
stats = df[['self_rank', 'score', 'percentile']].describe()
print(stats)


# 挑选需要重点审计的样本
print("\n--- Critical Samples to Audit (Low Rank & High Score) ---")
# 审计：Rank 很小但 Score 很大，说明 EIF 非常确定地找到了来源
critical_samples = df.sort_values(by=['self_rank'], ascending=[True]).head(10)
print(critical_samples[['query_index', 'self_rank', 'score']])

# 挑选需要重点审计的样本
print("\n--- Good Samples to Viz (Low Rank & High Score) ---")
# 审计：Rank 很小但 Score 很大，说明 EIF 非常确定地找到了来源
critical_samples = df.sort_values(by=['self_rank'], ascending=[False]).head(10)
print(critical_samples[['query_index', 'self_rank', 'score']])



# --- 3. 打印精美的统计报告 ---
print("-" * 40)
print(f"{'Metric':<20} | {'Success Rate':<15}")
print("-" * 40)
print(f"{'Top 1% (Rank>=' + str(int(full_rank*0.99)) + ')':<20} | {hit_top_1_percent:>13.2f}%")
print(f"{'Top 5% (Rank>=' + str(int(full_rank*0.95)) + ')':<20} | {hit_top_5_percent:>13.2f}%")
print(f"{'Top 10% (Rank>=' + str(int(full_rank*0.9)) + ')':<20} | {hit_top_10_percent:>13.2f}%")
print("-" * 40)
