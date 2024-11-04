#!/bin/bash

sudo apt install python3-pip -y

# setup distrifuser environment
pip install -e .
pip install accelerate

# setup docker environment
sudo apt-get update -y
sudo apt-get install ca-certificates curl -y
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
 
# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y

# setup clipscore
if [ -d "clipscore" ]; then
    echo "clipscore already exists, skipping setup."
else
    git clone https://github.com/jmhessel/clipscore.git
    cd clipscore
    pip install -r requirements.txt
fi

pip install qdrant-client

echo -e "\n0. Add your user to the docker group \n sudo usermod -aG docker $USER"
echo -e "\n1. Exit and login in again."
echo -e "\n2. Run the adrant service. \n docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant"
