#!/usr/bin/env bash
# Parallel live-OMNeT inference launcher (one slot per terminal).
# 1) Run configure_omnet_ports.py once if INI ports are not assigned yet.
# 2) In terminal k: start OMNeT for slot k, then run the matching COMMAND line.

# --- Slot 1 | TraCI port 9999 | gwu-workspace-pedestrians ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 1
eval "$COMMAND"

# --- Slot 2 | TraCI port 10001 | gwu-workspace-pedestrians-2 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-2/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 2
eval "$COMMAND"

# --- Slot 3 | TraCI port 10003 | gwu-workspace-pedestrians-3 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-3/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 3
eval "$COMMAND"

# --- Slot 4 | TraCI port 10005 | gwu-workspace-pedestrians-4 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-4/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 4
eval "$COMMAND"

# --- Slot 5 | TraCI port 10007 | gwu-workspace-pedestrians-5 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-5/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 5
eval "$COMMAND"

# --- Slot 6 | TraCI port 10009 | gwu-workspace-pedestrians-6 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-6/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 6
eval "$COMMAND"

# --- Slot 7 | TraCI port 10011 | gwu-workspace-pedestrians-7 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-7/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 7
eval "$COMMAND"

# --- Slot 8 | TraCI port 10013 | gwu-workspace-pedestrians-8 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-8/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 8
eval "$COMMAND"

# --- Slot 9 | TraCI port 10015 | gwu-workspace-pedestrians-9 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-9/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 9
eval "$COMMAND"

# --- Slot 10 | TraCI port 10017 | gwu-workspace-pedestrians-10 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-10/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 10
eval "$COMMAND"

# --- Slot 11 | TraCI port 10019 | gwu-workspace-pedestrians-11 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-11/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 11
eval "$COMMAND"

# --- Slot 12 | TraCI port 10021 | gwu-workspace-pedestrians-12 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-12/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 12
eval "$COMMAND"

# --- Slot 13 | TraCI port 10023 | gwu-workspace-pedestrians-13 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-13/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 13
eval "$COMMAND"

# --- Slot 14 | TraCI port 10025 | gwu-workspace-pedestrians-14 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-14/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 14
eval "$COMMAND"

# --- Slot 15 | TraCI port 10027 | gwu-workspace-pedestrians-15 ---
# OMNeT: cd /home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians-15/simu5G/simulations/NR/cars && opp_run ...
COMMAND=python3 experiments/run_inference_sweep.py --agent presslight --network hangzhou --train_pr 1.0 --rates 0.05 0.1 0.5 1.0 --checkpoint_selection topk --repeats 3 --live_omnet --omnet_slot 15
eval "$COMMAND"

