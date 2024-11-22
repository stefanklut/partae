from pathlib import Path

gerr_path = Path("~/jsons/gerr").expanduser().glob("*/")
gerr_path = set(path.name for path in gerr_path if path.is_dir())
checked_path = Path("~/jsons-checked").expanduser().glob("*/")
checked_path = set(path.name for path in checked_path if path.is_dir())
assert gerr_path, "No gerr directories found"
assert checked_path, "No checked directories found"

duplicates = gerr_path.intersection(checked_path)

print(duplicates)
