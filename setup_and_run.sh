#!/bin/bash

echo "[+] Creating virtual environment using Python 3.11..."
python3 -m venv venv

echo "[+] Activating virtual environment..."
source venv/bin/activate

echo "[+] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[+] Running the biosensor project..."
python main.py