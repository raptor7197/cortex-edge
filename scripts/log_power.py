import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""INA219/INA226 power logging for energy experiments (PLAN 15.14).

Designed for the Raspberry Pi with a 5 V INA219 breakout on I2C.
On machines without the sensor it explains what is needed and exits.

Usage: python scripts/log_power.py [--interval 0.1] [--out experiments/results/power.csv]

Net energy per query = query energy - (idle power x query duration).
"""

import argparse
import csv
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--out", default="experiments/results/power.csv")
    args = parser.parse_args()

    try:
        import board
        import busio
        from adafruit_ina219 import INA219
    except ImportError:
        raise SystemExit(
            "INA219 support not installed. On the Raspberry Pi:\n"
            "  pip install adafruit-circuitpython-ina219\n"
            "and connect the INA219 on I2C (SCL/SDA, address 0x40).\n"
            "On other machines power logging is unavailable."
        )

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = INA219(i2c)
    except Exception as exc:
        raise SystemExit(f"Sensor not found: {exc}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    energy_j, previous = 0.0, time.perf_counter()
    start = previous

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "voltage_v", "current_a", "power_w", "energy_j"])
        print(f"Logging power to {out} — Ctrl+C to stop.")
        while True:
            now = time.perf_counter()
            dt = now - previous
            previous = now
            voltage = sensor.bus_voltage + sensor.shunt_voltage
            current = sensor.current / 1000.0
            power = voltage * current
            energy_j += power * dt
            writer.writerow([round(now - start, 3), round(voltage, 3),
                             round(current, 4), round(power, 4), round(energy_j, 4)])
            f.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()