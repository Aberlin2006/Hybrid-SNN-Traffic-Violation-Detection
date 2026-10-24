# Hybrid-SNN: Traffic Violation Detection

## Overview

Hybrid-SNN is a hybrid computer vision and Spiking Neural Network (SNN) project for traffic violation classification, targeting the Microchip PolarFire SoC FPGA.

The system combines YOLOv8n object detection, image preprocessing, Time-To-First-Spike (TTFS) encoding, and Leaky Integrate-and-Fire (LIF) neurons.

## Architecture

1. Camera input and object detection using YOLOv8n.
2. Region-of-interest extraction and resizing to 48 × 48 RGB.
3. Image preprocessing using CLAHE and Difference of Gaussians (DoG).
4. TTFS spike encoding over eight time steps.
5. SNN classification using 64 hidden LIF neurons and 10 output classes.
6. Violation decision based on the predicted class.

## Technology Stack

- Python and PyTorch
- YOLOv8n
- Spiking Neural Networks
- TTFS encoding and LIF neurons
- Verilog RTL
- Microchip Libero SoC

## Reported Results

The project's evaluation materials report:

- Top-1 accuracy: 52.9%
- Top-5 accuracy: 94.5%
- Violation precision: 81.7%
- Violation recall: 83.6%
- Violation F1-score: 82.7%

These results should be reproduced using the corresponding evaluation code and model checkpoint before being treated as independently verified.

## Hardware Target

Microchip PolarFire SoC FPGA (MPFS250T).

Hardware latency, resource utilization, power consumption, and physical FPGA execution require confirmation from the corresponding implementation reports and board tests.

## Project Status

This repository documents the Hybrid-SNN project. The software pipeline, trained model, and FPGA implementation must be checked against their actual source files and validation reports before deployment claims are made.
