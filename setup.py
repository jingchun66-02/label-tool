from setuptools import setup, find_packages

setup(
    name='labeler_tool',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'Pillow',
        'click',
        'tqdm',
        'numpy',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'labeler=labeler_tool.cli:main',
        ],
    },
)
