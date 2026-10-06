"""Download licensed, labelled photos of roasted coffee-bean groups."""

from __future__ import annotations

import urllib.parse
import urllib.request
import urllib.error
import json
import time
from pathlib import Path


OUTPUT = Path("datasets/real-groups")
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# split by source photograph so crops from one photo never cross splits.
IMAGES = (
    ("Light", "train", "385 degrees cinnamon roast coffee.png"),
    ("Light", "val", "400 degrees new england roast coffee.png"),
    ("Light", "test", "410 degrees american roast coffee.png"),
    ("Medium", "train", "425 degrees city roast coffee.png"),
    ("Medium", "val", "440 degrees full city roast coffee.png"),
    ("Medium", "train", "Medium roasted Arabica coffee beans.jpg"),
    ("Medium", "test", "Peaberry coffee beans, close up.jpg"),
    ("Medium", "test", "Maragogipe Coffee.jpg"),
    ("Dark", "train", "450 degrees vienna roast coffee.png"),
    ("Dark", "val", "460 degrees french roast coffee.png"),
    ("Dark", "test", "470 degrees italian roast coffee.png"),
    ("Dark", "train", "480 degrees spanish roast coffee.png"),
    ("Dark", "train", "Dark roasted espresso blend coffee beans 1.jpg"),
    ("Dark", "test", "Dark roasted espresso blend coffee beans 2.jpg"),
)


def main() -> None:
    titles = "|".join(f"File:{filename}" for _, _, filename in IMAGES)
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "1600",
            "titles": titles,
        }
    )
    api_request = urllib.request.Request(
        f"{COMMONS_API}?{query}", headers={"User-Agent": "CoffeeRoastClassifier/1.0"}
    )
    with urllib.request.urlopen(api_request, timeout=60) as response:
        pages = json.load(response)["query"]["pages"].values()
    urls = {
        page["title"].removeprefix("File:"): page["imageinfo"][0].get(
            "thumburl", page["imageinfo"][0]["url"]
        )
        for page in pages
    }

    for class_name, split, filename in IMAGES:
        destination = OUTPUT / split / class_name / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(f"exists: {destination}")
            continue
        request = urllib.request.Request(
            urls[filename], headers={"User-Agent": "CoffeeRoastClassifier/1.0"}
        )
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    destination.write_bytes(response.read())
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 5:
                    raise
                delay = 8 * (attempt + 1)
                print(f"rate limited; retrying {filename} in {delay}s")
                time.sleep(delay)
        print(f"downloaded: {destination}")
        time.sleep(1)


if __name__ == "__main__":
    main()
