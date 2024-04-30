"""
Coders Strike Back Single mode environment
author: Alessandro Nicolosi
Heavily modified by: Kevin James (some of the original code was wacky)
url: https://github.com/alenic/gym-coders-strike-back
Original game: https://www.codingame.com/multiplayer/bot-programming/coders-strike-back
"""
import math
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np
import os



import math

class Vec:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, (int, float)):  # Multiplication by a scalar
            return Vec(self.x * other, self.y * other)
        raise NotImplementedError("Can only multiply Vec by a scalar")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):  # Division by a scalar
            return Vec(self.x / other, self.y / other)
        raise NotImplementedError("Can only divide Vec by a scalar")

    def __str__(self):
        return f"({self.x}, {self.y})"

    def to_tuple(self):
        return (self.x, self.y)

    def round(self):
        self.x = round(self.x)
        self.y = round(self.y)
        return self

    def int(self):
        self.x = int(self.x)
        self.y = int(self.y)
        return self

    def sqr_norm(self):
        return self.x**2 + self.y**2

def get_sqr_distance(v1,v2):
    x_diff = v1.x - v2.x
    y_diff = v1.y - v2.y
    return x_diff**2 + y_diff**2

def get_distance(v1, v2):
    return math.sqrt(get_sqr_distance(v1,v2))

def get_angle(v1, v2):
    x_diff = v2.x - v1.x
    y_diff = v2.y - v1.y
    return math.atan2(y_diff, x_diff)

def normalize_angle(angle):
    if angle > math.pi:
        angle -= 2 * math.pi  # Turn left
    elif angle < -math.pi:
        angle += 2 * math.pi  # Turn right
    return angle

def angle_to_vector(angle):
    return Vec(math.cos(angle), math.sin(angle))



class CodersStrikeBackBase():
    metadata = {"render.modes": ["human"], "video.frames_per_second": 30}

    def __init__(self):
        # Game constants
        self.max_checkpoints = 5
        self.max_thrust = 100.0
        self.maxSteeringAngle = 0.1 * np.pi
        self.friction = 0.85
        self.gamePixelWidth = 16000.
        self.gamePixelHeight = 9000.
        self.checkpoint_radius = 600
        self.pod_radius = 400
        self.checkpoint_radius_sqr = self.checkpoint_radius**2

        self.viewer = None
        self.n_laps = 3


    def getDeltaAngle(self, target):
        angle_to_target = get_angle(self.pos, target)
        delta_angle = angle_to_target - self.theta
        return normalize_angle(delta_angle)


    def checkpointCollision(self, checkpoint_pos):
        return self.collided(checkpoint_pos, self.checkpoint_radius)


    def collided(self, target_pos, target_radius):
        to_target = target_pos - self.pos
        if to_target.sqr_norm() < target_radius**2:
            return True

        move_vec = self.pos - self.pos_prev
        prev_to_target = target_pos - self.pos_prev

        # Check if target is not between previous and current positions
        if prev_to_target.sqr_norm() >= move_vec.sqr_norm():
            return False

        # Check if target is behind the movement direction
        dot_product = move_vec.dot(prev_to_target)
        if dot_product < 0:
            return False

        # Compute the closest point to the target on the movement path
        projection = dot_product / move_vec.sqr_norm()
        closest_point = self.pos_prev + move_vec * projection

        # Check if the closest point is within the target radius
        return get_sqr_distance(closest_point, target_pos) < target_radius**2


    # sample checkpoints with a minimum distance between them
    def sample_checkpoints(self, n):
        checkpoints = []
        while len(checkpoints) < n:
            checkpoint = self.gen_checkpoint()
            if self.valid_checkpoint(checkpoint, checkpoints):
                checkpoints.append(checkpoint)
        return checkpoints


    def gen_checkpoint(self):
        xmin = 0+self.checkpoint_radius/2
        ymin = 0+self.checkpoint_radius/2
        xmax = self.gamePixelWidth-self.checkpoint_radius/2
        ymax = self.gamePixelHeight-self.checkpoint_radius/2
        x = np.random.randint(xmin,xmax)
        y = np.random.randint(ymin,ymax)
        return Vec(x, y)


    def valid_checkpoint(self,checkpoint, checkpoints):
        for existing_check in checkpoints:
            if get_sqr_distance(checkpoint, existing_check) < (3*self.checkpoint_radius)**2:
                return False
        return True


    def reset(self):
        self.n_checkpoints = np.random.randint(3,self.max_checkpoints + 1)
        self.checkpoints = self.sample_checkpoints(self.n_checkpoints)
        self.theta = get_angle(self.checkpoints[-1], self.checkpoints[0])
        self.pos = self.checkpoints[-1]
        self.pos_prev = None
        self.vel = Vec(0,0)
        self.checkpoint_index = 0
        self.n_laps_remaining = 3

        self.steps_since_last_ckpt = 0
        self.time = 0
        self.viewer = None
        self.done = False
        self.completed = False
        self.failed = False


    def next_checkpoint(self):
        return self.checkpoints[self.checkpoint_index]


    def movePod(self, target, thrust):
        da = self.getDeltaAngle(target)
        da = np.clip(da, -self.maxSteeringAngle, self.maxSteeringAngle)

        self.theta = normalize_angle(self.theta + da)

        # Update dynamics
        prev_vel = self.vel
        self.vel += thrust * angle_to_vector(self.theta)
        self.pos = (self.pos + self.vel).round()

        self.vel = self.friction*self.vel.int()

    def get_targets(self):
        targets = []
        n = self.n_checkpoints
        cur_ind = self.checkpoint_index
        for i in range(self.max_checkpoints):
            targets.append(self.checkpoints[(i+cur_ind) % n])
        return targets


    def step(self, target, thrust):
        if self.done:
            return
        self.time += 1
        self.pos_prev = self.pos
        self.movePod(target, thrust)
        self.steps_since_last_ckpt += 1

        if self.checkpointCollision(self.next_checkpoint()):
            self.steps_since_last_ckpt = 0
            self.checkpoint_index += 1
            if self.checkpoint_index >= self.n_checkpoints:
                self.n_laps_remaining -= 1
                self.checkpoint_index -= self.n_checkpoints
                if self.n_laps_remaining <= 0:
                    self.done = True
                    self.completed = True

        if self.steps_since_last_ckpt >= 100:
            self.done = True
            self.failed = True


    def render(self, mode="human"):
        # Must be 16:9
        screen_width = 640
        screen_height = 360
        scale = screen_width / self.gamePixelWidth
        pod_diam = scale * self.pod_radius * 2.0
        checkpoint_diam = scale * self.checkpoint_radius * 2.0

        if self.viewer is None:
            import pygame_rendering
            self.viewer = pygame_rendering.Viewer(screen_width, screen_height)

            dirname = os.path.dirname(__file__)
            backImgPath = os.path.join(dirname, "imgs", "back.png")
            self.viewer.setBackground(backImgPath)

            ckptImgPath = backImgPath = os.path.join(dirname, "imgs", "ckpt.png")

            self.checkpointCircle = []
            for i in range(self.n_checkpoints):
                if i == self.n_checkpoints - 1:
                    display_num = "End"
                else:
                    display_num = i+1
                ckpt = scale * self.checkpoints[i]
                ckptObject = pygame_rendering.Checkpoint(
                    ckptImgPath,
                    pos=ckpt.to_tuple(),
                    number=display_num,
                    width=checkpoint_diam,
                    height=checkpoint_diam,
                )
                ckptObject.setVisible(False)
                self.viewer.addCheckpoint(ckptObject)

            podImgPath = backImgPath = os.path.join(dirname, "imgs", "pod.png")
            pod = scale * self.pos
            podObject = pygame_rendering.Pod(
                podImgPath,
                pos=pod.to_tuple(),
                theta=self.theta,
                width=pod_diam,
                height=pod_diam,
            )
            self.viewer.addPod(podObject)

            text = pygame_rendering.Text(
                "Time", backgroundColor=(0, 0, 0), pos=(0, 0)
            )
            self.viewer.addText(text)

        for ckpt in self.viewer.checkpoints:
            ckpt.setVisible(True)

        self.viewer.pods[0].setPos((scale*self.pos).to_tuple())
        self.viewer.pods[0].rotate(self.theta)

        self.viewer.text.setText(f'Time: {self.time}  Lap: {1+self.n_laps - self.n_laps_remaining}/{self.n_laps}')
        return self.viewer.render()


    def close(self):
        if self.viewer:
            self.viewer.close()



class CodersStrikeBackSingle(CodersStrikeBackBase, gym.Env):
    def __init__(self, seed=None):
        gym.Env.__init__(self)
        CodersStrikeBackBase.__init__(self)

        min_pos = -200000.0
        max_pos = 200000.0
        min_vel = -2000.0
        max_vel = 2000.0
        screen_max = [self.gamePixelWidth, self.gamePixelHeight]
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, min_pos, min_pos, min_vel, min_vel]+[0,0]*self.max_checkpoints),
            high=np.array([self.n_laps, np.pi, max_pos, max_pos, max_vel, max_vel]+screen_max*self.max_checkpoints),
            dtype=np.float64
        )

        self.action_space = spaces.Box(
            low = np.array([min_pos, min_pos, 0.0]),
            high = np.array([max_pos, max_pos, self.max_thrust]),
            dtype=np.float64
        )

    def get_observation(self):
        obs = [self.n_laps_remaining, self.theta,
               self.pos.x, self.pos.y,
               self.vel.x, self.vel.y,
               ]
        for t in self.get_targets():
            obs += [t.x, t.y]
        return np.array(obs)

    def reset(self, seed=None, options=None):
        super().reset()
        return self.get_observation(), {}

    def step(self, action):
        target = Vec(action[0], action[1])
        thrust = action[2]
        super().step(target,thrust)
        return self.get_observation(), self.reward(), self.done, self.done, {}

    def render(self):
        return super().render()

    def close(self):
        if self.viewer:
            self.viewer.close()

    def evaluation_reward(self):
        return -1 + 10000*int(self.completed) - 10000*int(self.failed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Put in your own reward function here
    def reward(self):
        return self.evaluation_reward()
