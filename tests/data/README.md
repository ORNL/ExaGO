## Scalability Testing

### Utilities for creating test cases

This directory contains:
- [`ieee-118-bus_v1.raw`](ieee-118-bus_v1.raw): original IEEE 118-bus case without modifications in PSS/E format,
- [`transform_ieee118_raw_to_100bus_m.py`](transform_ieee118_raw_to_100bus_m.py): script to convert power flow 118-bus case to a 100-bus _optimal_ power flow case in Matpower format,
- [`scale_network.py`](scale_network.py): script to assemble a network of arbitrary size.
