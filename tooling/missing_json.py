from pathlib import Path

seen_inventory = set()

json_path = "/data/data/cats-lessons/segregation"

for path in Path(json_path).expanduser().rglob("*.json"):
    inventory = path.parent.name
    if inventory not in seen_inventory:
        image_paths = list(Path(f"/data/spinque-converted/{inventory}/").rglob("*.jp2"))

        for image_path in image_paths:
            find_json = list(Path(json_path).expanduser().rglob(f"{image_path.stem}.json"))
            if not find_json:
                print(f"Missing json for {image_path}")
        seen_inventory.add(inventory)

    image_path = Path("/data/spinque-converted/").joinpath(path.parent.name, path.stem + ".jp2")
    if not image_path.exists():
        print(f"Missing image for {path}")
