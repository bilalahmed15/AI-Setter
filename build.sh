#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Updating package lists..."
apt-get update

echo "Installing PortAudio..."
apt-get install -y portaudio19-dev

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
