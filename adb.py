import os
import time
import subprocess
from datetime import datetime

# ------------------------------
# Step 1: Ensure required modules are installed
# ------------------------------
required_packages = ["uiautomator2", "pillow"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"[!] {package} not found. Installing...")
        subprocess.check_call(["pip", "install", package])

# ------------------------------
# Step 2: Import after install
# ------------------------------
import uiautomator2 as u2

# ------------------------------
# Step 3: Screenshot + scroll logic
# ------------------------------
SAVE_DIR = "screenshots_pokemon_cards"
os.makedirs(SAVE_DIR, exist_ok=True)

def capture_cards():
    print("\n[*] Connecting to device...")
    d = u2.connect()

    print("[*] Launching Pokémon TCG app...")
    package_name = "jp.pokemon.pokemontcgp"
    d.app_start(package_name)
    time.sleep(10)

    # Try to find and click "My Cards"
    print("[*] Locating 'My Cards'...")

    d.dump_hierarchy()  # refresh UI
    found = False

    if d(text="My Cards").exists:
        d(text="My Cards").click()
        found = True
    elif d(description="My Cards").exists:
        d(description="My Cards").click()
        found = True
    else:
        print("[!] 'My Cards' not found automatically.")
        input(">>> Please navigate manually to 'My Cards' and press Enter...")

    if found:
        print("[✓] Navigated to 'My Cards'")
        time.sleep(3)

    scroll_count = 10
    for i in range(scroll_count):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_path = f"/sdcard/pokecard_{i}_{timestamp}.png"

        d.screenshot(remote_path)
        print(f"[+] Screenshot saved on device: {remote_path}")

        os.system(f"adb pull {remote_path} {SAVE_DIR}")
        print(f"    → Saved to PC: {SAVE_DIR}/pokecard_{i}_{timestamp}.png")

        d.shell(["rm", remote_path])
        d.swipe(0.5, 0.8, 0.5, 0.3, 0.5)
        time.sleep(2)

    print("✅ Done capturing screenshots.")

# ------------------------------
# Step 4: Schedule every 3 hours
# ------------------------------
while True:
    print(f"\n=== Starting capture at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    try:
        capture_cards()
    except Exception as e:
        print(f"[!] Error: {e}")
    print("[*] Sleeping for 3 hours...\n")
    time.sleep(3 * 60 * 60)  # 3 hours
