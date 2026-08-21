import pickle
import matplotlib.pyplot as plt


try:
    with open("training_stats.pkl", "rb") as file:
        stats = pickle.load(file)
except FileNotFoundError:
    print("Error: Could not find training_stats.pkl.")
    exit()

rewards = stats['rewards']
steps = stats['steps']
events = stats['events']
rounds = range(1, len(rewards) + 1)


coins_collected = [rnd_dict['COIN_COLLECTED'] for rnd_dict in events]
invalid_actions = [rnd_dict['INVALID_ACTION'] for rnd_dict in events]


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))


ax1.plot(rounds, rewards, color='blue', alpha=0.7)
ax1.set_title('Total Reward per Round')
ax1.set_xlabel('Training Round')
ax1.set_ylabel('Reward')
ax1.grid(True, linestyle='--', alpha=0.5)


ax2.hist(steps, bins=30, color='orange', edgecolor='black', alpha=0.7)
ax2.set_title('Time Taken to Finish (Distribution)')
ax2.set_xlabel('Steps per Round')
ax2.set_ylabel('Frequency')
ax2.grid(True, linestyle='--', alpha=0.5)

ax3.plot(rounds, coins_collected, label='Coins Collected', color='green', alpha=0.8)
ax3.plot(rounds, invalid_actions, label='Wall Crashes (Invalid)', color='red', alpha=0.8)
ax3.set_title('Behavior Breakdown per Round')
ax3.set_xlabel('Training Round')
ax3.set_ylabel('Number of Occurrences')
ax3.legend()
plt.title("Coin-greedy agent trained with SARSA and decaying epsilon")
ax3.grid(True, linestyle='--', alpha=0.5)


plt.tight_layout()
plt.savefig("Coin_greedy_SARSA_Epsilon_decay.png", dpi=300)
plt.show()
