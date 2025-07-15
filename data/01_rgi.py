# Copyright (C) 2025 Andy Aschwanden
#
# This file is part of pism-terra.
#
# PISM-TERRA is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-TERRA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Prepare RGI.
"""
# pylint: disable=unused-import,assignment-from-none,unexpected-keyword-arg

from pathlib import Path
from pism_terra.download import download_archive

baseurl = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-C/RGI2000-v7.0-C-RGIREGION.zip"

path = Path("rgi")
path.mkdir(parents=True, exist_ok=True)

region = "01_alaska"
url = f"https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-C/RGI2000-v7.0-C-{region}.zip"
archive = download_archive(url)
archive.extractall(path=path)


wgs84 = "EPSG:4326"

url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/World_UTM_Grid/FeatureServer/0/query?where=1=1&outFields=*&f=geojson"
utm_zones = gpd.read_file(url)

rgi = gpd.read_file("data/rgi/RGI2000-v7.0-C-01_alaska.shp")
# Extract bounds from original glacier polygons
bounds = rgi.geometry.bounds.reset_index(drop=True)

# Rename columns for clarity
bounds.columns = ["minx", "miny", "maxx", "maxy"]

# Add bounds to rgi_with_utm
rgi_with_bounds = rgi.reset_index(drop=True).join(bounds)

rgi_center = rgi.copy()
rgi_center["geometry"] = gpd.points_from_xy(rgi["cenlon"], rgi["cenlat"])
rgi_points = gpd.GeoDataFrame(rgi_center, geometry="geometry", crs=rgi.crs).to_crs(wgs84)

# Ensure UTM grid is in WGS84
utm_zones = utm_zones.to_crs(wgs84)

# Spatial join: assign each centroid the matching UTM zone polygon attributes
rgi_with_utm = gpd.sjoin(rgi_points, utm_zones[["ZONE", "ROW_", "geometry"]], how="left", predicate="within")

# Rename for clarity (optional)
rgi_with_utm = rgi_with_utm.rename(columns={"ZONE": "utm_zone_assigned", "ROW_": "utm_row"})

rgi_with_utm = gpd.GeoDataFrame(rgi_with_utm, geometry=rgi.geometry, crs=rgi.crs).to_crs(wgs84)
# Preview
print(rgi_with_utm[["rgi_id", "cenlon", "cenlat", "utm_zone_assigned", "utm_row"]])
