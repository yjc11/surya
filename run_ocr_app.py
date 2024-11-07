import argparse
import os
import subprocess


def run_app():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_app_path = os.path.join(cur_dir, "ocr_app.py")
    cmd = ["streamlit", "run", ocr_app_path, '--server.port', '9199', '--server.address', '0.0.0.0']
    subprocess.run(cmd, env={**os.environ, "IN_STREAMLIT": "true"})


if __name__ == "__main__":
    run_app()
