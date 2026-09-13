
from pathlib import Path


base_dir = Path(__file__).resolve().parent
source_path = base_dir / "week2_eda_source.txt"
output_path = base_dir / "week2_eda.py"

if source_path.exists():
    output_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Written {output_path.stat().st_size} bytes to {output_path}")
elif output_path.exists():
    print(f"Source file not found; keeping existing {output_path}")
else:
    raise FileNotFoundError(f"Neither {source_path} nor {output_path} exists")
