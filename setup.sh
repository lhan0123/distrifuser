#!/bin/bash

# setup distrifuser environment
pip install -e .

# setup clipscore
git clone https://github.com/jmhessel/clipscore.git
cd clipscore
pip install -r requirements.txt