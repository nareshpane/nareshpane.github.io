# Standalone qgis2web Export

This folder contains the local JavaScript, vector data, styles, fonts, and legend assets needed by the Leaflet application. The OpenStreetMap basemap remains an external network dependency.

Do not open `index.html` through a `file://` URL. OpenStreetMap can reject those tile requests because they lack the expected HTTP context. Serve the directory locally instead:

```bash
cd "research/GIS Learning/web/qgis2web"
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Publishing on a Website

Upload the entire folder to a web server so that all relative data, script, CSS, font, image, and legend paths remain intact. The map requires internet access for OpenStreetMap tiles.

Example:

`https://www.example.com/my-map/index.html`

The map displays the required TransLink data legend, OpenStreetMap contributor attribution, and qgis2web/Leaflet/QGIS credits. See `../../ATTRIBUTION.md`, `../../DATA_LICENSE.md`, and `THIRD_PARTY_NOTICES.md` before redistribution.
