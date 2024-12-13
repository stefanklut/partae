import json
from pathlib import Path

from natsort import natsorted

json_path = "~/jsons-checked"


# Loop over all json files
for path in natsorted(list(Path(json_path).expanduser().rglob("*.json"))):
    with path.open(mode="r") as f:
        # Check if more than 1 line
        lines = f.readlines()

    # If more than 1 line, print the path and update the json file
    if len(lines) > 1:
        print(f"More than 1 line in {path}")
        with path.open(mode="r") as f:
            data = json.load(f)
            scan_id = data["scanId"]
            is_first_page = data["isFirstPage"]
            user = "stefank"

        with path.open(mode="w") as f:
            json.dump({"scanId": scan_id, "isFirstPage": is_first_page, "user": user}, f)
