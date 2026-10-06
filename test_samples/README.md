# Manual test images

These files are held-out source photographs and were not used to train the
deployed ensemble. Do not move them into a training split unless they are also
replaced with new independent test photographs.

| File | Expected class | Source | License |
| --- | --- | --- | --- |
| `light-roast.jpg` | Light | Crop of [410 degrees American roast coffee](https://commons.wikimedia.org/wiki/File:410_degrees_american_roast_coffee.png), Dan Bollinger | CC BY-SA 3.0 |
| `medium-roast.jpg` | Medium | [Peaberry coffee beans, close up](https://commons.wikimedia.org/wiki/File:Peaberry_coffee_beans,_close_up.jpg), Sage Ross | CC BY-SA 3.0 / GFDL |
| `dark-roast.png` | Dark | [470 degrees Italian roast coffee](https://commons.wikimedia.org/wiki/File:470_degrees_italian_roast_coffee.png), Dan Bollinger | CC BY-SA 3.0 |

The Light image is a resized crop selected from the held-out source. The
Medium and Dark filenames were changed for easier manual testing; their image
pixels are otherwise unmodified. These are functional smoke-test examples,
not a substitute for the complete held-out accuracy report.
