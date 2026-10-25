| Method | Accuracy (%) | Macro-F1 (%) | P_d (%) | P_fa (%) | P_d, JSR < −4 dB (%) |
|---|---|---|---|---|---|
| RadarNet-1D (FP32) | 76.43 | 76.73 | 91.64 | 25.87 | 83.45 |
| RadarNet-1D INT8 (hardware, bit-exact) | 76.26 | 76.58 | 91.40 | 25.85 | 83.07 |
| Expert features + GBM | 75.51 | 76.01 | 90.36 | 24.65 | 79.92 |
| Energy detector | – | – | 91.31 | 25.42 | 66.77 |

Per-class F1 (%) — Normal: 66.1, Barrage: 77.6, Tone: 80.1, Sweep: 90.2, Pulsed: 85.1, Replay: 61.9, CoChannel: 76.0
