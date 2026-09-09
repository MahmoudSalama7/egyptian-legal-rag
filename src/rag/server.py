from __future__ import annotations

import os
import subprocess
import sys

from dotenv import load_dotenv


def main():
    """Reads .env configuration and launches MLflow UI with SQL backend automatically."""
    load_dotenv()
    db_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    port = os.getenv("MLFLOW_PORT", "5000")

    print(
        f"[MLflow Server] Launching MLflow UI with Backend Store: {db_uri} on Port: {port}..."
    )
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        db_uri,
        "--port",
        str(port),
    ]
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n[MLflow Server] MLflow UI stopped cleanly.")


if __name__ == "__main__":
    main()
