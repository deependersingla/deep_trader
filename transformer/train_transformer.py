"""
Training script for the Transformer-based DQN Trading Agent.

Pipeline:
    1. Load EURUSD M1 data via StockDataLoader
    2. Chronological split: Train (2012-2020), Val (2021-2022), Test (2023-2025)
    3. Train Transformer DQN with Double DQN + PER
    4. Periodic validation & CSV logging
    5. Save weights to transformer/saved_networks/
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# tensor-reinforcement uses a hyphen → not a valid Python package name.
_tr_dir = os.path.join(os.path.dirname(__file__), '..', 'tensor-reinforcement')
if _tr_dir not in sys.path:
    sys.path.insert(0, _tr_dir)
from modern_data import StockDataLoader

from transformer.transformer_env import SequenceTradingEnv
from transformer.transformer_agent import TransformerDQNAgent

# ======================================================================
# Paths
# ======================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'EURUSD_M1_2012_2025.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_networks')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'training_results_transformer.csv')

# ======================================================================
# GPU Setup
# ======================================================================
print("\n" + "=" * 60)
print("  🔧  Hardware Check")
print("=" * 60)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  ✅ GPU Detected: {len(gpus)} device(s)")
        print(f"     Device Name: {gpus[0].name}")
    except RuntimeError as e:
        print(f"  ❌ GPU Error: {e}")
else:
    print("  ⚠️  No GPU detected — training will run on CPU.")
    print("     Install tensorflow[and-cuda] for GPU acceleration.")
print("=" * 60 + "\n")


# ======================================================================
# Evaluation helper
# ======================================================================
def evaluate(agent: TransformerDQNAgent, env: SequenceTradingEnv,
             episodes: int = 3) -> tuple[float, float]:
    """Run evaluation episodes with exploration disabled."""
    total_rewards, total_profits = [], []
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(episodes):
        obs, _ = env.reset()
        agent.reset_sequence()
        episode_reward = 0.0
        done = False

        while not done:
            # The env already returns sequences; the agent still needs
            # the *single* latest state to maintain its own buffer, but
            # we can simply pass the last row of the observation.
            single_state = obs[-1]
            action = agent.act(single_state)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_profits.append(info["total_value"] - env.initial_balance)

    agent.epsilon = saved_epsilon
    return float(np.mean(total_rewards)), float(np.mean(total_profits))


# ======================================================================
# Main
# ======================================================================
def main():
    # ---- Hyperparameters (tuned for M1 forex data) ----
    SEQ_LEN = 64            # ~1 hour of M1 candles as context
    EPISODE_LENGTH = 1000   # ~2-3 trading days per episode
    EPISODES = 50           # Training episodes (increase for production)
    EVAL_EVERY = 5          # Validate every N episodes
    SAVE_EVERY = 5          # Save weights every N episodes

    # 1. Load Data ─────────────────────────────────────────────────────
    print("📂 Loading Data...")
    loader = StockDataLoader(DATA_PATH)
    try:
        loader.load_csv()
        loader.preprocess()
    except Exception as e:
        print(f"❌ Data Loading Error: {e}")
        return

    full_state_data, full_price_data, full_dates = loader.get_data()

    # 2. Chronological Split ───────────────────────────────────────────
    print("\n📊 Splitting Data Chronologically...")

    train_mask = full_dates < np.datetime64('2021-01-01')
    val_mask   = (full_dates >= np.datetime64('2021-01-01')) & (full_dates < np.datetime64('2023-01-01'))
    test_mask  = full_dates >= np.datetime64('2023-01-01')

    splits = {
        "Train": train_mask,
        "Val":   val_mask,
        "Test":  test_mask,
    }

    for name, mask in splits.items():
        n = mask.sum()
        if n == 0:
            print(f"❌ {name} split is empty! Check date ranges.")
            return
        d0 = pd.to_datetime(str(full_dates[mask][0]))
        d1 = pd.to_datetime(str(full_dates[mask][-1]))
        print(f"   {name:>5s}: {n:>10,} samples  ({d0:%Y-%m-%d} → {d1:%Y-%m-%d})")

    train_states, train_prices = full_state_data[train_mask], full_price_data[train_mask]
    val_states,   val_prices   = full_state_data[val_mask],   full_price_data[val_mask]

    # 3. Environments ──────────────────────────────────────────────────
    env     = SequenceTradingEnv(train_states, train_prices, seq_len=SEQ_LEN, episode_length=EPISODE_LENGTH)
    val_env = SequenceTradingEnv(val_states,   val_prices,   seq_len=SEQ_LEN, episode_length=EPISODE_LENGTH)

    feature_dim = train_states.shape[1]
    action_dim  = int(env.action_space.n)

    # 4. Agent ─────────────────────────────────────────────────────────
    agent = TransformerDQNAgent(
        feature_dim=feature_dim,
        action_dim=action_dim,
        seq_len=SEQ_LEN,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dff=128,
        dropout_rate=0.1,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        batch_size=32,
        memory_size=50_000,
        target_update_freq=500,
    )

    # 5. Logging ───────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write("Episode,Train_Reward,Val_Reward,Val_Profit,Epsilon\n")

    # 6. Training Loop ─────────────────────────────────────────────────
    print(f"\n🚀 Starting Training ({EPISODES} episodes) — Logging to {LOG_FILE}")
    print("-" * 70)

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        agent.reset_sequence()
        total_reward = 0.0

        while True:
            single_state = obs[-1]  # last row of the sequence
            action = agent.act(single_state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Agent stores sequences internally via remember()
            agent.remember(single_state, action, reward, next_obs[-1], done)

            obs = next_obs
            total_reward += reward

            # Train
            agent.replay()

            if done:
                profit = info["total_value"] - env.initial_balance
                print(
                    f"  Episode {ep:>3d}/{EPISODES} │ "
                    f"Reward: {total_reward:>10.2f} │ "
                    f"Profit: {profit:>10.2f} │ "
                    f"ε: {agent.epsilon:.4f}"
                )

                # Periodic evaluation
                if ep % EVAL_EVERY == 0:
                    val_reward, val_profit = evaluate(agent, val_env, episodes=3)
                    print(
                        f"       ↳ 🔎 Validation │ "
                        f"Avg Reward: {val_reward:>10.2f} │ "
                        f"Avg Profit: {val_profit:>10.2f}"
                    )
                    with open(LOG_FILE, "a") as f:
                        f.write(f"{ep},{total_reward},{val_reward},{val_profit},{agent.epsilon}\n")

                # Save weights
                if ep % SAVE_EVERY == 0:
                    path = os.path.join(MODELS_DIR, f"transformer_ep{ep}.weights.h5")
                    agent.save(path)

                break

    print("-" * 70)
    print("✅ Training Finished.")


# ======================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
