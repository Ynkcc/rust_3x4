import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv('training_log.csv')

# Display the first few rows and column info to understand the structure
print(df.head())
print(df.info())
# Set up the figure with subplots
fig, axes = plt.subplots(4, 2, figsize=(20, 24))
plt.subplots_adjust(hspace=0.3)

# 1. Loss Analysis
axes[0, 0].plot(df['iteration'], df['avg_total_loss'], label='Total Loss', color='purple')
axes[0, 0].plot(df['iteration'], df['avg_policy_loss'], label='Policy Loss', color='blue')
axes[0, 0].plot(df['iteration'], df['avg_value_loss'], label='Value Loss', color='orange')
axes[0, 0].set_title('Training Losses')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Entropy and Confidence
axes[0, 1].plot(df['iteration'], df['avg_policy_entropy'], label='Policy Entropy', color='green')
axes[0, 1].set_ylabel('Entropy', color='green')
axes[0, 1].tick_params(axis='y', labelcolor='green')
ax2 = axes[0, 1].twinx()
ax2.plot(df['iteration'], df['high_confidence_ratio'], label='High Confidence Ratio', color='red', linestyle='--')
ax2.set_ylabel('High Confidence Ratio', color='red')
ax2.tick_params(axis='y', labelcolor='red')
axes[0, 1].set_title('Policy Entropy & Confidence')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].grid(True)

# 3. Game Outcomes (Win/Draw/Loss)
axes[1, 0].stackplot(df['iteration'], df['red_win_ratio'], df['draw_ratio'], df['black_win_ratio'],
                     labels=['Red Win', 'Draw', 'Black Win'], colors=['#ff9999', '#cccccc', '#666666'], alpha=0.7)
axes[1, 0].set_title('Game Outcome Ratios')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Ratio')
axes[1, 0].legend(loc='upper left')
axes[1, 0].grid(True)

# 4. Average Game Steps
axes[1, 1].plot(df['iteration'], df['avg_game_steps'], label='Avg Game Steps', color='brown')
axes[1, 1].set_title('Average Game Steps')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Steps')
axes[1, 1].grid(True)

# 5. Scenario 1 Analysis (Value & Moves)
axes[2, 0].plot(df['iteration'], df['scenario1_value'], label='Scenario 1 Value', color='black', linewidth=2)
axes[2, 0].plot(df['iteration'], df['scenario1_masked_a38'], label='Prob Move A38', linestyle='--')
axes[2, 0].plot(df['iteration'], df['scenario1_masked_a39'], label='Prob Move A39', linestyle='--')
axes[2, 0].plot(df['iteration'], df['scenario1_masked_a40'], label='Prob Move A40', linestyle='--')
axes[2, 0].set_title('Scenario 1: Value & Move Probabilities')
axes[2, 0].set_xlabel('Iteration')
axes[2, 0].legend()
axes[2, 0].grid(True)

# 6. Scenario 2 Analysis (Value & Moves)
axes[2, 1].plot(df['iteration'], df['scenario2_value'], label='Scenario 2 Value', color='black', linewidth=2)
axes[2, 1].plot(df['iteration'], df['scenario2_masked_a3'], label='Prob Move A3', linestyle='--')
axes[2, 1].plot(df['iteration'], df['scenario2_masked_a5'], label='Prob Move A5', linestyle='--')
axes[2, 1].set_title('Scenario 2: Value & Move Probabilities')
axes[2, 1].set_xlabel('Iteration')
axes[2, 1].legend()
axes[2, 1].grid(True)

# 7. Sample Efficiency
axes[3, 0].plot(df['iteration'], df['replay_buffer_size'], label='Buffer Size', color='teal')
axes[3, 0].set_title('Replay Buffer Size')
axes[3, 0].set_xlabel('Iteration')
axes[3, 0].grid(True)

# 8. New Samples Count
axes[3, 1].plot(df['iteration'], df['new_samples_count'], label='New Samples', color='magenta')
axes[3, 1].set_title('New Samples Generated per Iteration')
axes[3, 1].set_xlabel('Iteration')
axes[3, 1].grid(True)

plt.savefig('alphazero_analysis.png')