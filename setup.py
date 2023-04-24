from setuptools import setup
from setuptools.command.install import install
import subprocess
import os

with open("requirements.txt") as f:
      PACKAGES=f.read().splitlines()

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

setup(name='arctic_ai',
      version='0.1',
      description='Python package for Arctic Workflow. Mirrors jupyter developments.',
      url='https://github.com/jlevy44/ArcticAI_Prototype',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=[],
      entry_points={
            'console_scripts':['arctic_ai=arctic_ai.cli:main']
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['arctic_ai'],
      install_requires=PACKAGES)
