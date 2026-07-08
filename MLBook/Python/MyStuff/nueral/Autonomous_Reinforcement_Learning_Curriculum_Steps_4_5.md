TURNED INTO A NOTEBOOOK




# Autonomous Reinforcement Learning Curriculum: Steps 4 and 5

This document captures the lesson content for Step 4 and Step 5 of the autonomous reinforcement learning curriculum. It focuses on game interfacing, computer vision preprocessing, virtual input control, Deep Q-Network design, experience replay, and target-network stabilization.

## Step 4 Curriculum: The Unabridged Game Bridge

### Step 4 Masterclass: The Unabridged Game Bridge (Vision, Input, and Pipeline Engineering)

Welcome to the guide for building a standalone hardware and software bridge in Python. The goal is to interface directly with operating system buffers, virtual device drivers, and pixel matrices without relying on high-level wrappers.

### Module 0: The Pythonic Structural Shift for Lua Developers

Before building a high-throughput pipeline, review how Python data structures differ from Lua:

| Concept | Lua Mechanics | Python Mechanics | Production Gotcha |
| --- | --- | --- | --- |
| Memory tables | Single data type (`{}`), dynamically resizes hash tables | Specialized types: lists `[]` and tuples `()` | Tuples are immutable |
| Matrix storage | Nested tables with 1-based indexing | Dense contiguous blocks via NumPy arrays, 0-indexed | Slicing arrays uses `[y, x]`, not `[x, y]` |
| Concurrency | Coroutines, single-threaded | Native threads and multiprocessing | The GIL limits CPU-bound multi-threading |

### Module 1: Dynamic Optical Target Hooking

In a standalone game, the window can move, resize, minimize, or be obscured. Hardcoding absolute coordinates is fragile. A resilient bridge should dynamically locate the target window, read its bounds, and grab frames in real time.

#### Documentation and Research Blueprint

Install the window management and capture dependencies:

```bash
pip install mss opencv-python numpy pygetwindow
```

Research the following interfaces:

1. PyGetWindow layouts: study `pygetwindow.getWindowsWithTitle()` and review returned properties such as `top`, `left`, `width`, `height`, and `isActive`.
2. OS focus handling: understand what happens if a game window is minimized and whether a capture tool can still read pixel data from an inactive or hidden window.

#### Practice Challenge 1: The Window Targeter

Objective: write a script that finds an application by title, tracks its coordinates, and builds an MSS bounding box that follows the window as it moves.

```python
import time
import pygetwindow as gw
import mss
import numpy as np
import cv2

TARGET_TITLE = "Notepad"

print(f"Searching for active window matching title: '{TARGET_TITLE}'...")

matching_windows = gw.getWindowsWithTitle(TARGET_TITLE)

if not matching_windows:
    print(f"Error: Could not find any windows matching '{TARGET_TITLE}'")
    exit()

game_window = matching_windows[0]
print(f"Target Found! Initial Geometry: Left={game_window.left}, Top={game_window.top}")

sct = mss.mss()

while True:
    monitor_region = {
        "top": game_window.top,
        "left": game_window.left,
        "width": game_window.width,
        "height": game_window.height,
    }

    screenshot = sct.grab(monitor_region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cv2.imshow("Dynamic AI Vision Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### Module 2: Image Matrix Preprocessing

High-resolution frames are expensive and noisy. A reinforcement learning pipeline should downsample them into compact grayscale matrices, usually around 84 by 84 pixels.

#### Documentation and Research Blueprint

Open the OpenCV documentation and research:

1. `cv2.cvtColor()` flags for converting BGR to grayscale.
2. `cv2.resize()` interpolation modes, especially which shrinking algorithm is best for aggressive downsampling.

#### Practice Challenge 2: The Data Cruncher

Objective: capture a screen region and immediately compress it into an 84 by 84 grayscale matrix.

```python
import cv2
import mss
import numpy as np
import time

sct = mss.mss()

capture_region = {"top": 100, "left": 100, "width": 800, "height": 600}
print("Running Preprocessing Engine...")

while True:
    raw_img = sct.grab(capture_region)
    frame = np.array(raw_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ai_ready_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)

    if ai_ready_frame is not None:
        print(f"Matrix Dimension Profile: {ai_ready_frame.shape}", end="\r")

    cv2.imshow("What Humans See (Raw)", frame)
    cv2.imshow("What the Neural Network Sees (Preprocessed)", ai_ready_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### Module 3: Virtual Controller Drive Execution

Games often ignore standard OS window messages and use low-level controller APIs. To interact safely, use a virtual gamepad library such as `vgamepad`.

#### Documentation and Research Blueprint

Install the virtualization dependency:

```bash
pip install vgamepad
```

Research the following:

1. `left_joystick_float()` and its supported range.
2. `right_trigger_float()` and how trigger values map from unpressed to fully pressed.

#### Practice Challenge 3: The Precision Joystick Sweep

Objective: create a virtual gamepad, wait for safety, and perform a smooth steering sweep from left to right.

```python
import time
import vgamepad as vg

gamepad = vg.VX360Gamepad()
print("Virtual Device registered on host OS bus.")
print("Take the next 5 seconds to open a gamepad tester website or click into your game...")
time.sleep(5)
print("Starting smooth analog steering sweep...")

for percent in range(-100, 101, 5):
    steering_vector = percent / 100.0
    gamepad.left_joystick_float(x_value_float=steering_vector, y_value_float=0.0)
    gamepad.update()
    print(f"Simulating Hardware Vector: Left Stick X = {steering_vector:<5}", end="\r")
    time.sleep(0.05)

print("\nCleaning up controller state...")
gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
gamepad.update()
print("Hardware Sweep Complete.")
```

### Capstone Framework Construction

Combine all three modules into a single production script that captures the target window, preprocesses frames, and drives the virtual controller.

```python
import time
import cv2
import mss
import numpy as np
import pygetwindow as gw
import vgamepad as vg

TARGET_WINDOW_NAME = "Trackmania"
AI_INPUT_RESOLUTION = (84, 84)

print("Initializing Unified Game Bridge Framework...")

windows = gw.getWindowsWithTitle(TARGET_WINDOW_NAME)
if not windows:
    print(f"Initialization Failed: Could not locate window title '{TARGET_WINDOW_NAME}'")
    exit()

game_win = windows[0]
sct = mss.mss()
gamepad = vg.VX360Gamepad()

print("Pipeline successfully coupled. Focus your game window now.")
time.sleep(3)

last_execution_time = time.time()

try:
    while True:
        bounds = {
            "top": game_win.top,
            "left": game_win.left,
            "width": game_win.width,
            "height": game_win.height,
        }

        raw_capture = sct.grab(bounds)
        raw_matrix = np.array(raw_capture)
        raw_matrix = cv2.cvtColor(raw_matrix, cv2.COLOR_BGRA2BGR)

        ai_state = cv2.cvtColor(raw_matrix, cv2.COLOR_BGR2GRAY)
        ai_state = cv2.resize(ai_state, AI_INPUT_RESOLUTION, interpolation=cv2.INTER_AREA)

        current_time = time.time()
        loop_latency = current_time - last_execution_time
        last_execution_time = current_time
        fps = 1.0 / loop_latency if loop_latency > 0 else 0

        print(f"BRIDGE HEALTH | Pipeline FPS: {int(fps):<3} | State Matrix: {str(ai_state.shape):<10}", end="\r")

        if ai_state is not None:
            cv2.imshow("Bridge Core State View", ai_state)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nUser manually interrupted processing.")

finally:
    print("\nSafely uncoupling virtual hardware bus devices...")
    gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gamepad.update()
    cv2.destroyAllWindows()
    print("Bridge gracefully shutdown.")
```

## Step 5 Curriculum: Deep Q-Networks with Visual State Architectures

### Step 5 Masterclass: Deep Q-Networks with Image States

#### Objective

Implement a Deep Q-Network using PyTorch. The model should process 84 by 84 grayscale frames, use a CNN feature extractor, store transitions in replay memory, and stabilize training with a target network.

### Module 1: The Lua-to-Python Tensor Map

PyTorch tensors require a specific layout and data type convention:

| Concept | Torch7 (Lua) | PyTorch (Python) | Production Impact |
| --- | --- | --- | --- |
| Dimension reshape | `tensor:view(B, C, H, W)` | `tensor.view(B, C, H, W)` or `tensor.reshape()` | Rearranging spatial dimensions may require `permute()` or `transpose()` |
| Device management | `tensor:cuda()` | `tensor.to(device)` | Explicitly move tensors to CPU or GPU |
| Gradient clearing | Handled manually | `optimizer.zero_grad()` | Leftover gradients can accumulate and destabilize learning |

### Module 2: The Curse of Dimensionality and the CNN Approximation

A Q-table is not practical for image-based states. Instead, use a convolutional neural network as a functional approximator.

#### Documentation and Research Blueprint

Install PyTorch:

```bash
pip install torch torchvision
```

Research:

1. `torch.nn.Conv2d` parameters such as `in_channels`, `out_channels`, `kernel_size`, `stride`, and `padding`.
2. `torch.nn.Linear` and `torch.nn.Flatten` for connecting convolutional features to action outputs.
3. The DQN architecture from the Nature 2015 paper by Mnih et al.

#### Practice Challenge 1: The CNN Q-Network

Objective: define a convolutional network that maps a single-channel image to action values.

```python
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, h, w, action_size):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, h, w)
            dummy_out = self.conv3(self.conv2(self.conv1(dummy_input)))
            flat_features_size = self.flatten(dummy_out).shape[1]

        self.fc1 = nn.Linear(flat_features_size, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values
```

### Module 3: The Experience Replay Buffer

Sequential frames are highly correlated, so training directly on live experience can overfit. A replay buffer breaks this dependency by sampling random historical transitions.

#### Documentation and Research Blueprint

Research `collections.deque` and `random.sample`.

#### Practice Challenge 2: The Memory Bank

Objective: build a replay buffer capable of storing transitions and sampling random minibatches.

```python
import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)
```

### Module 4: Stabilizing the Moving Target

The Bellman target changes while the model trains, so a target network is used to stabilize optimization.

#### Practice Challenge 3: The Deep Q-Loss Step

Objective: implement the training step that samples replay data, computes target Q-values, and performs backpropagation.

```python
import torch
import torch.optim as optim

policy_net = DQN(84, 84, action_size=3).to("cpu")
target_net = DQN(84, 84, action_size=3).to("cpu")
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)


def train_step(replay_buffer, batch_size, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    state_t = torch.FloatTensor(states).unsqueeze(1) / 255.0
    next_state_t = torch.FloatTensor(next_states).unsqueeze(1) / 255.0
    action_t = torch.LongTensor(actions).unsqueeze(1)
    reward_t = torch.FloatTensor(rewards).unsqueeze(1)
    done_t = torch.FloatTensor(dones).unsqueeze(1)

    current_q = policy_net(state_t).gather(1, action_t)

    with torch.no_grad():
        max_next_q = target_net(next_state_t).max(1)[0].unsqueeze(1)
        expected_q = reward_t + (gamma * max_next_q * (1 - done_t))

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(current_q, expected_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Ultimate Capstone Pipeline

The production pipeline combines the game bridge and the DQN loop:

1. Capture the target window dynamically.
2. Convert the frame into an 84 by 84 grayscale tensor.
3. Feed the state into the policy network.
4. Map the chosen action to virtual controller inputs.
5. Execute the game step and observe the next frame.
6. Store the transition in replay memory.
7. Sample a batch and update the policy network.
8. Periodically synchronize the target network.

This lesson provides the core structure needed for autonomous visual reinforcement learning systems.