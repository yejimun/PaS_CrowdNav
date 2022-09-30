from setuptools import setup


setup(
    name='CrowdNav_PaSRL',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.utils',
        'pytorchBaselines'
    ],
    install_requires=[
        'gitpython',
        'gym',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
