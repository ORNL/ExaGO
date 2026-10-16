"""Data preprocessing code."""

from pathlib import Path

import pandas as pd
from shapely import STRtree, from_wkt
from shapely.geometry import shape

TABLE_COLUMNS = {
    "transmission_lines": {
        "wkt": "str",
        "flow capacity": "float64",
        "pf": "float64",
        "qf": "float64",
        "pt": "float64",
        "qt": "float64",
        "kilovolt": "float64",
        "line_name": "str",
        "srouce": "str",
        "target": "str",
        "actual flow": "float64",
        "loading percent": "float64",
    },
    "buses": {
        "coordinates": "str",
        "bus_name": "str",
        "kilovolt levels": "str",
        "number of buses": "int64",
        "vm": "float64",
        "county": "str",
        "state": "str",
    },
    "generators": {
        "coordinates": "str",
        "Power generated": "float64",
        "Power capacity": "float64",
        "Remaining Capacity": "float64",
        "KVlevels": "str",
        "color": "str",
        "generation name": "str",
        "number of buses": "int64",
        "generation type": "str",
        "county": "str",
        "state": "str",
    },
}


def reference_tables(data_path, reference_data_path):
    """Load reference files, falling back to installed data if no local data exists."""
    tables = {}
    for table, filename in (("states", "us_states.csv"), ("counties", "counties.csv")):
        local_path = Path(data_path) / filename
        path = (
            local_path if local_path.is_file() else Path(reference_data_path) / filename
        )
        tables[table] = pd.read_csv(path)
    return tables


def network_tables(data, counties):
    """Construct DuckDB tables from case data."""
    features = data.get("geojsondata", {}).get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(
            "Choose an ExaGO JSON case containing geojsondata and network features."
        )

    county_index = STRtree(from_wkt(counties["WKT"].to_numpy()))
    rows = {table: [] for table in TABLE_COLUMNS}
    colors = {
        "wind": "green",
        "solar": "yellow",
        "coal": "gray",
        "nuclear": "red",
        "hydro": "blue",
        "ng": "orange",
    }

    for feature in features:
        p = feature["properties"]
        if p.get("start", 0) != 0:
            continue
        geometry = shape(feature["geometry"])
        if geometry.geom_type == "Point":
            county, state = None, None
            matches = county_index.query(geometry, predicate="within")
            if len(matches):
                match = counties.iloc[matches[0]]
                county, state = match["NAME"], match["states"]
            buses = p["bus"]
            rows["buses"].append(
                {
                    "coordinates": geometry.wkt,
                    "bus_name": p["NAME"],
                    "kilovolt levels": str(set(p["KVlevels"])),
                    "number of buses": p["nbus"],
                    "vm": sum(bus["VM"] for bus in buses) / len(buses),
                    "county": county,
                    "state": state,
                }
            )
            generators = [gen for bus in buses for gen in bus["gen"]]
            if generators:
                power = sum(gen["GEN_STATUS"] * gen["PG"] for gen in generators)
                capacity = sum(gen["GEN_STATUS"] * gen["PMAX"] for gen in generators)
                fuel = generators[-1]["GEN_FUEL"].lower()
                rows["generators"].append(
                    {
                        "coordinates": geometry.wkt,
                        "Power generated": power,
                        "Power capacity": capacity,
                        "Remaining Capacity": abs(capacity - power),
                        "KVlevels": str({bus["BASE_KV"] for bus in buses}),
                        "color": colors.get(fuel, "black"),
                        "generation name": p["NAME"],
                        "number of buses": p["nbus"],
                        "generation type": fuel,
                        "county": county,
                        "state": state,
                    }
                )
        elif geometry.geom_type == "LineString":
            source, target = p["NAME"].split(" -- ")
            if p["PF"] <= 0:
                source, target = target, source
            rows["transmission_lines"].append(
                {
                    "wkt": geometry.wkt,
                    "flow capacity": p["RATE_A"],
                    "pf": p["PF"],
                    "qf": p["QF"],
                    "pt": p["PT"],
                    "qt": p["QT"],
                    "kilovolt": p["KV"],
                    "line_name": p["NAME"],
                    "srouce": source,
                    "target": target,
                    "actual flow": abs(p["PF"]),
                    "loading percent": abs(p["PF"]) / p["RATE_A"] * 100
                    if p["RATE_A"] > 0
                    else None,
                }
            )

    return {
        table: pd.DataFrame(rows[table], columns=columns).astype(columns)
        for table, columns in TABLE_COLUMNS.items()
    }


def load_case(connection, data, references):
    """Populate a DuckDB database with case information."""
    tables = {**references, **network_tables(data, references["counties"])}
    for table, frame in tables.items():
        connection.register("case_data", frame)
        connection.execute(f'CREATE TABLE "{table}" AS SELECT * FROM case_data')
        connection.unregister("case_data")
