"""
AgroVision Deployment
Run from project root: python run.py
"""

import subprocess, sys, os, webbrowser, time, threading

PORT = 8000
URL  = f"http://localhost:{PORT}"

def open_browser():
    time.sleep(2.5)
    webbrowser.open(URL)

def main():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║       AgroVision Deployment Server       ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  UI   →  {URL:<32}║")
    print(f"║  Docs →  {URL + '/docs':<32}║")
    print("║  Press CTRL+C to stop                    ║")
    print("╚══════════════════════════════════════════╝")
    print()

    threading.Thread(target=open_browser, daemon=True).start()

    # Must run from project root so paths resolve correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.serve.app:app",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--reload"
    ])

if __name__ == "__main__":
    main()