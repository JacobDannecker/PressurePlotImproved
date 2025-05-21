#!/bin/bash
echo "Starting installation of python environment"
# Define the virtual environment name and packages to install
ENV_NAME="pimp_env"

sudo usermod -aG dialout $USER

# Copy all files to home dir
cp -r PIMP $HOME

cd $HOME/PIMP/

# Install python venv
sudo apt install python3-venv

# Create virtual environment 
python3 -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Upgrade pip and install packages
pip3 install --upgrade pip
#pip3 install "${PACKAGES[@]}"
pip3 install -r requirements.txt
# Deactivate the virtual environment
deactivate

chmod +x $HOME/PIMP/start.sh

# Ad alias to bash_aliases 
echo "alias pimp=\"$HOME/PIMP/start.sh\"" >> $HOME/.bash_aliases
source $HOME/.bashrc

echo "Installation complete. LOGOUT OF SYSTEM to make changes available! You can start PIMP by typing pimp."

