# debug_preprocess.py
import os, traceback
print("CURRENT WORKDIR:", os.getcwd())

# show config paths
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH
print("RAW_DATA_PATH =", RAW_DATA_PATH)
print("CLEAN_DATA_PATH =", CLEAN_DATA_PATH)

# show whether raw file exists and its size / first 3 lines
if os.path.exists(RAW_DATA_PATH):
    print("RAW file exists. Size (bytes):", os.path.getsize(RAW_DATA_PATH))
    try:
        with open(RAW_DATA_PATH, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"RAW_LINE_{i+1}:", line.strip())
    except Exception as e:
        print("Could not read RAW file:", e)
else:
    print("RAW file DOES NOT exist at RAW_DATA_PATH")

# run the preprocess function inside try/except to capture errors
try:
    from src.preprocess import preprocess_and_save
except Exception as e:
    print("Failed to import preprocess_and_save:", e)
    traceback.print_exc()
else:
    try:
        print("Calling preprocess_and_save() ...")
        preprocess_and_save()
        print("preprocess_and_save() finished normally.")
    except Exception as e:
        print("Exception during preprocess_and_save():", e)
        traceback.print_exc()

# finally list files in data folder
print("\nDATA folder listing:")
data_dir = os.path.join(os.path.dirname(__file__), "data")
try:
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        print("-", name, "(size:", os.path.getsize(p), "bytes)")
except Exception as e:
    print("Could not list data folder:", e)
