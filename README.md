<h1 align="center">Learning-Based Passive Fault-Tolerant Control of a Quadrotor with Rotor Failure</h1>

<div align="center">

[📄 Arxiv Paper](https://arxiv.org/abs/2503.02649) • [🎥 Video Demonstration](https://www.youtube.com/watch?v=9i4SaDhRscQ)

</div>

---

## 🚀 TODO

- [x] Release paper and demonstration video  
- [x] Release trained policy and simulation environment for validation  
- [ ] Release real-world implementation framework (based on PX4)  
- [ ] Release training pipeline  

---

## ✅ Validation Instructions

To validate the trained policy in simulation:

1. Clone this repository:
   ```bash
   git clone https://github.com/HITSZcjh/uav_ftc.git
   cd uav_ftc
   ```

2. Run the simulation script:
   ```bash
   python validation/main.py
   ```

> **Environment Requirements** (tested on Python 3.8):  
- `onnx`               1.17.0  
- `onnxruntime`        1.16.3  
- `matplotlib`         3.7.4  
- `numpy`              1.24.4

---
