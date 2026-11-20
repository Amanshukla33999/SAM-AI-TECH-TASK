"""Downloads the Kaggle dataset (gokulrajkmv/unemployment-in-india) into the data/ folder."""
import shutil
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

if list(DATA.glob("*.csv")):
    print("Dataset already present in data/ - skipping download.")
else:
    try:
        import kagglehub
        print("Downloading dataset from Kaggle ...")
        src = Path(kagglehub.dataset_download("gokulrajkmv/unemployment-in-india"))
        for f in src.glob("*.csv"):
            shutil.copy(f, DATA / f.name)
            print("Saved", f.name)
    except Exception as e:
        print("\nAuto-download failed:", e)
        print("Manual fix: download the CSV from")
        print("  https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india")
        print("and put it inside the 'data' folder. (You can also upload it inside the app.)")
