from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
print(env.action_space)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
print(env.action_space.n)
print(env.action_space.sample())

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(a)
    env.render()

env.close()