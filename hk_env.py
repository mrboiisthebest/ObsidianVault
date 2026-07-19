# My code but reformatted

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import time
import pydirectinput
from pathlib import Path

class HollowKnightGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, win, sct):
        super(HollowKnightGymEnv, self).__init__()
        
        self.win = win
        self.sct = sct
        
        # --- Constants ---
        self.TOTAL_MASKS = 9
        self.MATCH_SCALE = 0.5
        self.MATCH_THRESHOLD = 0.55
        
        self.lower_mask_pink = np.array([210, 205, 222])  # B, G, R
        self.upper_mask_pink = np.array([230, 222, 240])  # B, G, R
        
        self.HORNET_RED_LO_1 = np.array([170, 90, 70])  
        self.HORNET_RED_HI_1 = np.array([180, 255, 220]) 
        
        self.PLAYER_WHITE_LO = np.array([0, 0, 240])    
        self.PLAYER_WHITE_HI = np.array([180, 15, 255])
        
        self.left_held = False
        self.right_held = False

        # --- Load Player Templates ---
        print("Pre-loading player templates into RAM...", flush=True)
        self.player_templates = self._get_player_images(flip=True)
        
        # --- Gym Spaces ---
        # Action Space: 6 discrete choices
        self.action_space = spaces.Discrete(6)
        
        # Observation space vector
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([2560, 1440, 2560, 1440, 9, 3000], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # --- State Tracking Variables ---
        self.last_checked_mask = self.TOTAL_MASKS
        self.time_lost_boss = 0.0
        self.lost_boss = False
        self.last_seen_boss = time.time()
        self.last_time_stamp = time.time()
        self.last_attack_time = time.time()
        
        # Health down debounce setup
        self.hp_debounce_frames = 10
        self.current_hp_deb_frame = 0
        self.health_debounce = False

        # Input configuration
        pydirectinput.PAUSE = 0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Run automated game reset sequence
        self._reset_game_macro()
        
        # 2. Re-establish tracking baseline
        raw_frame = self._capture_screen()
        health_frame = raw_frame[80:120, 245:675, :3]
        
        self.last_checked_mask = self._get_masks(health_frame)
        if self.last_checked_mask == 0:
            self.last_checked_mask = self.TOTAL_MASKS
            
        self.lost_boss = False
        self.last_seen_boss = time.time()
        self.last_time_stamp = time.time()
        
        # 3. Pull initial state observations
        obs = self._get_observation(raw_frame)
        info = {}
        return obs, info

    def step(self, action):
        # 1. Execute action
        self._take_action(action)
        
        # 2. Gather tracking updates
        raw_frame = self._capture_screen()
        
        processed_frame = cv2.resize(raw_frame, (128, 90), interpolation=cv2.INTER_LINEAR)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2GRAY)
        brightness = np.mean(processed_frame)
        
        boss_box = self._get_boss_bounds(raw_frame)
        player_box = self._get_player_bounds(raw_frame, boss_bounds=boss_box)
        
        boss_hit_detected = False
        
        health_frame = raw_frame[80:120, 245:675, :3]
        current_masks = self._get_masks(health_frame)
        
        # 3. Handle Health Debounce Logic
        if self.current_hp_deb_frame < self.hp_debounce_frames and not self.health_debounce:
            self.current_hp_deb_frame += 1
        elif self.current_hp_deb_frame >= self.hp_debounce_frames:
            self.current_hp_deb_frame = 0
            self.health_debounce = True
            
        if self.last_checked_mask < current_masks:
            self.last_checked_mask = current_masks

        # 4. Check Event Conditions (Hit / Death / Loading)
        boss_hit_detected = self._check_boss_hit()
        
        # Determine if taken damage
        damage_taken = False
        if self.last_checked_mask > current_masks and brightness < 235 and self.health_debounce:
            amount_down = self.last_checked_mask - current_masks
            if amount_down <= 2:
                damage_taken = True
                self.last_checked_mask = current_masks

        # 5. Compile Rewards
        reward = 0.0
        terminated = False
        truncated = False
        
        # Small passive reward for staying alive
        reward += 0.02
        
        if boss_hit_detected:
            reward += 20.0
            print("RL Event: Hit Boss! Reward +20", flush=True)
            
        if damage_taken:
            reward -= 15.0
            print(f"RL Event: Damage Taken! Masks: {current_masks}. Reward -15", flush=True)
            
        # Terminal State Checks
        if current_masks <= 0 and brightness < 230:
            reward -= 50.0
            terminated = True
            print("RL Event: Player Died! Reward -50", flush=True)
            
        if brightness > 250:
            # Handle loading screen pauses without breaking steps
            time.sleep(3)

        # 6. Build current environment observation vector
        obs = self._build_obs_vector(player_box, boss_box, current_masks)
        
        return obs, reward, terminated, truncated, {}

    # --- Internal Vision & Spatial Adapters ---
    
    def _capture_screen(self):
        window_dims = {
            "left": self.win.left,
            "top": self.win.top,
            "width": self.win.width,
            "height": self.win.height
        }
        return np.array(self.sct.grab(window_dims))

    def _get_masks(self, health_frame):
        frame_height, frame_width, _ = health_frame.shape
        mask_width = health_frame.shape[1] // self.TOTAL_MASKS
        
        current_masks = 0
        for i in range(self.TOTAL_MASKS):
            start_x = i * mask_width
            end_x = (i + 1) * mask_width
            
            single_mask_box = health_frame[0:frame_height, start_x:end_x]
            isolated_color_box = cv2.inRange(single_mask_box, self.lower_mask_pink, self.upper_mask_pink)
            box_match_pixels = np.sum(isolated_color_box == 255)
            
            if box_match_pixels > 30: 
                current_masks += 1
        return current_masks

    def _get_boss_bounds(self, raw_frame):
        hsv_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.HORNET_RED_LO_1, self.HORNET_RED_HI_1)
        
        open_kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        
        close_kernel = np.ones((9, 9), np.uint8)
        final_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, close_kernel)
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if not self.lost_boss:
                self.last_time_stamp = time.time()
            self.lost_boss = True
            return None

        valid_boss_parts = [cnt for cnt in contours if cv2.contourArea(cnt) > 150]
        
        if not valid_boss_parts:
            if not self.lost_boss:
                self.last_time_stamp = time.time()
            self.lost_boss = True
            return None
        
        if self.lost_boss:
            self.time_lost_boss = time.time() - self.last_time_stamp
            self.lost_boss = False
            self.last_seen_boss = time.time()
        
        all_points = np.vstack(valid_boss_parts)
        x, y, w, h = cv2.boundingRect(all_points)
        return (x, y, w, h)

    def _get_player_bounds(self, raw_frame, boss_bounds=None):
        hsv_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.PLAYER_WHITE_LO, self.PLAYER_WHITE_HI)
        
        h, w = mask.shape
        small_w, small_h = max(1, int(w * self.MATCH_SCALE)), max(1, int(h * self.MATCH_SCALE))
        mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
        _, mask_small = cv2.threshold(mask_small, 127, 255, cv2.THRESH_BINARY)
        
        if boss_bounds is not None:
            bx, by, bw, bh = boss_bounds
            sbx, sby = int(bx * self.MATCH_SCALE), int(by * self.MATCH_SCALE)
            sbw, sbh = int(bw * self.MATCH_SCALE), int(bh * self.MATCH_SCALE)
            padding = int(15 * self.MATCH_SCALE + 25)
            
            cv2.rectangle(
                mask_small, 
                (max(0, sbx - padding), max(0, sby - padding)), 
                (min(mask_small.shape[1], sbx + sbw + padding), min(mask_small.shape[0], sby + sbh + padding)), 
                0, -1
            )
        
        cv2.rectangle(mask_small, (0, 0), (int(500 * self.MATCH_SCALE), int(200 * self.MATCH_SCALE)), 0, -1)

        best_max_val = -1
        best_max_loc = None
        best_w, best_h = 0, 0
        
        for template in self.player_templates:
            th, tw = template.shape 
            match_map = cv2.matchTemplate(mask_small, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match_map)
            
            if max_val > best_max_val:
                best_max_val = max_val
                best_max_loc = max_loc
                best_w, best_h = tw, th

        if best_max_val > self.MATCH_THRESHOLD:
            inv_scale = 1.0 / self.MATCH_SCALE
            hx = int(best_max_loc[0] * inv_scale)
            hy = int(best_max_loc[1] * inv_scale)
            tw = int(best_w * inv_scale)
            th = int(best_h * inv_scale)
            
            px = hx - 2
            py = hy
            pw = tw + 4
            ph = int(th * 1.5)
            ph = min(ph, raw_frame.shape[0] - py)
            return (px, py, pw, ph)
            
        return None

    def _get_player_images(self, flip=True, scale=0.5):
        folder = Path(r"C:\Users\Owner\Desktop\PythonStuff\ObsidianVault\images\hollowknight\player")
        images = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        if not folder.exists():
            return []
        for file_path in folder.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, thresh_t = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
                    variants = [thresh_t]
                    if flip:
                        variants.append(cv2.flip(thresh_t, 1))
                    for v in variants:
                        h, w = v.shape
                        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                        small = cv2.resize(v, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        _, small_binary = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
                        images.append(small_binary)
        return images

    def _check_boss_hit(self):
        if 0.20 < self.time_lost_boss < 1.00 and (time.time() - self.last_attack_time) < 1.00:
            self.time_lost_boss = 0.0 
            return True
        return False

    # --- Observation Processing ---
    
    def _get_observation(self, raw_frame):
        boss_box = self._get_boss_bounds(raw_frame)
        player_box = self._get_player_bounds(raw_frame, boss_bounds=boss_box)
        health_frame = raw_frame[80:120, 245:675, :3]
        masks = self._get_masks(health_frame)
        return self._build_obs_vector(player_box, boss_box, masks)

    def _build_obs_vector(self, p_box, b_box, masks):
        px, py = (p_box[0], p_box[1]) if p_box else (0, 0)
        bx, by = (b_box[0], b_box[1]) if b_box else (0, 0)
        dist = np.sqrt((px - bx)**2 + (py - by)**2)
        return np.array([px, py, bx, by, masks, dist], dtype=np.float32)

    # --- Key Actuators & Macros ---
    
    def _take_action(self, action):
        # 1. Clean up key hold logic states
        if action != 1 and self.left_held:
            pydirectinput.keyUp('a')
            self.left_held = False
        if action != 2 and self.right_held:
            pydirectinput.keyUp('d')
            self.right_held = False

        # 2. Translate discrete action directly to inputs
        if action == 0:  # Stay still
            pass
        elif action == 1:  # Move Left
            if not self.left_held:
                pydirectinput.keyDown('a')
                self.left_held = True
        elif action == 2:  # Move Right
            if not self.right_held:
                pydirectinput.keyDown('d')
                self.right_held = True
        elif action == 3:  # Attack
            pydirectinput.press('r') 
            self.last_attack_time = time.time()
        elif action == 4:  # Jump
            pydirectinput.press('space')
        elif action == 5:  # Dash
            pydirectinput.press('t')

    def _reset_game_macro(self):
        # Clear direction locks before macro navigation runs
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('d')
        time.sleep(5.0)
        
        print("[Macro] Resetting...")
        pydirectinput.press('space')
        time.sleep(2.0)
        pydirectinput.press('w')
        time.sleep(0.3)
        pydirectinput.press('space')
        time.sleep(3.0)