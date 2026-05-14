# wire-length-vision

Machine vision tool that measures wire lengths from a photograph of a wiring panel.

Point a camera square-on at the panel, include a mm ruler in the shot, and the tool
segments each wire by colour, traces its centreline skeleton, and reports lengths in mm.

## How it works

1. **Ruler calibration** — detects tick marks on the ruler via Hough line transform
   to establish a pixels-per-mm scale factor.  Falls back to an interactive
   click-two-points mode if auto-detection fails.
2. **Colour segmentation** — converts the image to HSV and applies per-colour range
   masks defined in `config/colour_ranges.yaml`.  Morphological cleanup and hole-filling
   handle specular highlights on shiny insulation.
3. **Skeleton tracing** — skeletonises each wire mask, builds a pixel graph, prunes
   spur artefacts, and sums all edge arc-lengths.
4. **Output** — annotated PNG with centreline overlays and length labels, plus
   JSON and CSV result files.

## Requirements

- Python 3.10+
- See `requirements.txt` (OpenCV, scikit-image, NetworkX, SciPy, NumPy, PyYAML, Pandas)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Basic run — ruler auto-detected across the whole image, 1 mm per tick
python main.py panel.jpg

# Specify the ruler strip as a region-of-interest (x y w h) and tick spacing
python main.py panel.jpg --roi 0 850 1200 60 --mm-per-tick 10

# Higher accuracy skeletonisation for thick cables
python main.py panel.jpg --precise

# Write results to a custom directory
python main.py panel.jpg --output results/job42/
```

Results are written to `output/` by default:

```
output/
  panel_20260514_143022_annotated.png   ← image with coloured centreline overlays
  panel_20260514_143022_results.json    ← full measurement data + calibration metadata
  panel_20260514_143022_results.csv     ← colour, wire_id, length_mm, flags
```

A summary table is also printed to the terminal:

```
Wire           Length   Flags
---------------------------------------------
blue_1         218.4 mm  ok
blue_2          95.0 mm  crossing_interpolated
red_1          342.1 mm  ok
black_1        401.2 mm  high_specular
```

## Tuning colours for your lighting

The default HSV ranges in `config/colour_ranges.yaml` work under neutral white light.
For different lighting or unusual wire colours, sample ranges interactively:

```bash
python setup_colours.py panel.jpg --write
```

Click on each wire in the window, type the colour name in the terminal, then press **Q**.
The config file is updated in-place.

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config/colour_ranges.yaml` | HSV colour range config |
| `--output` | `output/` | Directory for result files |
| `--mm-per-tick` | `1.0` | Ruler graduation spacing in mm |
| `--min-area` | `500` | Minimum wire blob area in pixels (filters noise) |
| `--roi X Y W H` | *(whole image)* | Crop to this region for ruler detection |
| `--precise` | off | Use `medial_axis` skeleton (slower, better on thick wires) |
| `--no-clahe` | off | Disable CLAHE brightness normalisation |

## Known limitations

- **Wire crossings** of the same colour cannot be separated without depth information;
  the measurement is flagged `crossing_interpolated`.
- **Perspective** — the camera must be roughly square-on.  For panels photographed at
  an angle, supply `--roi` pointing to a rectangular reference and consider applying a
  homography warp manually before processing.
- **Reflective insulation** is handled partially; very shiny wires are flagged
  `high_specular` and may read slightly short.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
