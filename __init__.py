"""CPIoT-XAD2025 - hybrid CNN/LSTM cross-domain anomaly detection.

Forecast-based anomaly detection on a row-aligned fusion of network-telemetry
and physical-process features, under a leakage-safe, anomaly-mass-stratified
evaluation protocol.

Submodules that do not require TensorFlow (``config``, ``data``, ``splitting``,
``windowing``, ``scoring``, ``metrics``, ``residual_head``) can be imported on
their own, which is what the test suite does.
"""

__version__ = "1.0.0"
