from setuptools import setup


setup(
    name='PaS_CrowdNav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.utils',
        'rl'
    ],
)
