"""Read the result back from a saved figure.

``fig.save()`` embeds every number and text (not the raw data) in the file.
Data: mtcars from R's datasets package. Writes: weight_mpg.svg, weight_mpg.png
"""

import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image  # Pillow is installed with matplotlib

import inferplot as ip

df = pd.DataFrame(
    {
        "wt": [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440],
        "mpg": [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2],
    }
)
fig = ip.scatterstats(df, "wt", "mpg")
fig.save("weight_mpg.svg")
fig.save("weight_mpg.png")

# SVG: the Dublin Core description in the file's metadata.
description = "{http://purl.org/dc/elements/1.1/}description"
svg_json = ET.parse("weight_mpg.svg").getroot().find(f".//{description}").text
# PNG: the "Description" text chunk. PDF keeps it as the document's "Subject".
with Image.open("weight_mpg.png") as image:
    png_json = image.text["Description"]

result = ip.StatsResult.from_json(svg_json)
print(result.text.apa)
print(result.text.apa == fig.result.text.apa, svg_json == png_json)
print(len(result.values), result.check() == fig.result.check())
