#!/bin/bash
set -euo pipefail

read -rp "Enter mode (generated or unknown): " MODE
read -rp "Enter alpha in (0,2): " ALPHA
read -rp "Enter beta in (1,10): " BETA
read -rp "Enter gamma in (1,20): " GAMMA
read -rp "Enter Experiment id: " ID

python3 main.py --mode "$MODE" --alpha "$ALPHA" --beta "$BETA" --gamma "$GAMMA" --id "$ID"
