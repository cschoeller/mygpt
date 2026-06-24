from pathlib import Path
import shutil

import kagglehub

data_dir = Path(__file__).parent.resolve()
output_path = data_dir / "vadimkurochkin/wikitext-103"
output_path.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download("vadimkurochkin/wikitext-103", output_dir=str(output_path), force_download=False)

print("Path to dataset files:", path)

# Move wikitext-103 up to the data directory
dest_path = data_dir / "wikitext-103"
if not dest_path.exists():
    shutil.move(str(output_path), str(dest_path))
    print(f"Moved wikitext-103 to: {dest_path}")
else:
    print(f"Destination already exists, skipping move: {dest_path}")

# Delete the now-empty vadimkurochkin folder
vadimkurochkin_dir = data_dir / "vadimkurochkin"
if vadimkurochkin_dir.exists():
    shutil.rmtree(str(vadimkurochkin_dir))
    print(f"Deleted folder: {vadimkurochkin_dir}")
