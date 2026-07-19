import pywinctl
import mss
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from hk_env import HollowKnightGymEnv

# 1. Initialize Window Tracking Assets
TARGET_WINDOW = "Hollow Knight"
windows = pywinctl.getWindowsWithTitle(TARGET_WINDOW)
if not windows:
    print("Error: Target window missing!", flush=True)
    exit()

win = windows[0]
sct = mss.mss()

# 2. Instantiate and Wrap Your Custom Environment
# The Monitor wrapper is REQUIRED so the callback can count your lives/deaths
base_env = HollowKnightGymEnv(win=win, sct=sct)
env = Monitor(base_env)

# 3. Set Your Custom Life/Epoch Limit
# If you set this to 100, the script cuts off right as the 100th run ends!
MAX_BOSS_RUNS = 10
stop_callback = StopTrainingOnMaxEpisodes(max_episodes=MAX_BOSS_RUNS, verbose=1)

# 4. Create the PPO Model Agent Brain
# NOTE: Check the policy toggle below based on what data you feed the network!
model = PPO(
    "MlpPolicy",   # Keep "MlpPolicy" for vector coordinates. Change to "CnnPolicy" if passing raw pixels!
    env, 
    verbose=1, 
    learning_rate=0.0003,
    tensorboard_log="./tensorboard_hk/"
)

# 5. Fire Off the Learning Cycle
# We give it a massive total_timesteps pool so it only stops when your callback limit hits
print(f"Starting RL Agent Training Run for exactly {MAX_BOSS_RUNS} runs...", flush=True)
model.learn(total_timesteps=1000000, callback=stop_callback)

# 6. Save the trained weight network
model.save(f"hk_hornet_slayer_{MAX_BOSS_RUNS}_runs")
print("Saved models successfully!", flush=True)