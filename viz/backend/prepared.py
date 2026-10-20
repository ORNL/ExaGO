"""Browser-independent map data preparation and validation."""

import copy
import math

from shapely import STRtree
from shapely.geometry import MultiPoint, mapping, shape

STATE_ABBR = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
    "60": "AS",
    "66": "GU",
    "69": "MP",
    "72": "PR",
    "78": "VI",
}
COLORS = {
    "wind": "green",
    "solar": "yellow",
    "coal": "gray",
    "nuclear": "red",
    "hydro": "blue",
    "ng": "orange",
}


def collection(features):
    return {"type": "FeatureCollection", "features": features}


def is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_case(data):
    """Perform network validation."""
    geo = data.get("geojsondata") if isinstance(data, dict) else None
    if (
        not isinstance(geo, dict)
        or geo.get("type") != "FeatureCollection"
        or not geo.get("features")
    ):
        raise ValueError(
            "Choose an ExaGO JSON output containing geojsondata and network features."
        )
    if not isinstance(geo["features"], list):
        raise TypeError("Network features must be a list.")
    for feature in geo["features"]:
        if not isinstance(feature, dict):
            raise TypeError("Every network feature must have geometry and properties.")
        p, geometry = feature.get("properties"), feature.get("geometry")
        if not isinstance(p, dict) or not isinstance(p.get("NAME"), str):
            raise TypeError("Every network feature must have a name and properties.")
        if not isinstance(geometry, dict) or geometry.get("type") not in (
            "Point",
            "LineString",
        ):
            raise ValueError("Network features must be points or transmission lines.")
        coords = (
            [geometry.get("coordinates")]
            if geometry["type"] == "Point"
            else geometry.get("coordinates")
        )
        if (
            not isinstance(coords, list)
            or not coords
            or any(
                not isinstance(c, list)
                or len(c) < 2
                or not all(is_number(v) for v in c[:2])
                or abs(c[1]) >= 90
                for c in coords
            )
        ):
            raise ValueError(
                "Network coordinates must contain valid longitude and latitude values."
            )
        if geometry["type"] == "Point" and (
            not isinstance(p.get("KVlevels"), list)
            or not isinstance(p.get("bus"), list)
            or not p["bus"]
            or any(
                not isinstance(bus, dict) or not is_number(bus.get("VM"))
                for bus in p["bus"]
            )
        ):
            raise ValueError(
                "Each substation must contain bus voltages and voltage levels."
            )


class MapReferences:
    def __init__(self, counties):
        self.counties = counties["features"]
        self.index = STRtree([shape(f["geometry"]) for f in self.counties])


def hulls(data, key):
    features = []
    for value in data.get(key, []):
        coords = [
            f["geometry"]["coordinates"][:2]
            for f in data["geojsondata"]["features"]
            if f["geometry"]["type"] == "Point"
            and f["properties"].get(key[:-1]) == value
        ]
        if len(coords) < 3:
            continue
        hull = MultiPoint(coords).convex_hull
        if hull.geom_type == "Polygon":
            features.append(
                {
                    "type": "Feature",
                    "properties": {"name": value},
                    "geometry": mapping(hull),
                }
            )
    return collection(features)


def prepare_map(source, references):
    validate_case(source)
    data = copy.deepcopy(source)
    grid = collection(
        [
            f
            for f in data["geojsondata"]["features"]
            if f["properties"].get("start", 0) == 0
        ]
    )
    if not grid["features"]:
        raise ValueError("The case has no network features in its first time slice.")
    points = [f for f in grid["features"] if f["geometry"]["type"] == "Point"]
    counties = {}
    for point in points:
        p, buses = point["properties"], point["properties"]["bus"]
        p.update(
            Vm=sum(b["VM"] for b in buses) / len(buses),
            Va=buses[0].get("VA"),
            Pd=sum(b.get("PD", 0) for b in buses),
            Qd=sum(b.get("QD", 0) for b in buses),
            bus_type=buses[0].get("BUS_TYPE"),
            countyname="",
        )
        for index in sorted(
            references.index.query(shape(point["geometry"]), predicate="covered_by")
        ):
            index = int(index)
            if index not in counties:
                county = copy.deepcopy(references.counties[index])
                cp = county["properties"]
                cp.update(Pd=0, Vm_avg=0, KVlevels=[])
                cp["countyname"] = (
                    f"{cp['NAME']}, {STATE_ABBR.get(cp['STATE'], cp['STATE'])}"
                )
                counties[index] = [county, 0]
            county, count = counties[index]
            cp = county["properties"]
            p["countyname"] = cp["countyname"]
            cp["KVlevels"].extend(p["KVlevels"])
            cp["Pd"] += sum(b.get("PDloss", 0) for b in buses)
            cp["Vm_avg"] += sum(b["VM"] for b in buses)
            counties[index][1] = count + len(buses)
    county_features = []
    for index in sorted(counties):
        county, count = counties[index]
        county["properties"]["Vm_avg"] /= count
        county_features.append(county)
    countyload = collection(county_features)
    max_pd = max([0] + [c["properties"]["Pd"] for c in county_features])

    locations = [
        {
            "id": p["properties"]["NAME"],
            "name": p["properties"]["NAME"],
            "lon": p["geometry"]["coordinates"][0],
            "lat": p["geometry"]["coordinates"][1],
        }
        for p in points
    ]
    ids = {p["id"] for p in locations}
    flows, reactive = [], []
    for feature in grid["features"]:
        if feature["geometry"]["type"] != "LineString":
            continue
        p = feature["properties"]
        ends = p["NAME"].split(" -- ")
        if len(ends) != 2 or any(end not in ids for end in ends):
            continue
        a, b = ends
        rating = p["RATE_A"] if p.get("RATE_A", 0) > 0 else 10000
        for field, records in (("PF", flows), ("QF", reactive)):
            value = p.get(field)
            if is_number(value) and is_number(p.get("KV")):
                records.append(
                    {
                        "origin": a if value >= 0 else b,
                        "dest": b if value >= 0 else a,
                        "count": p["KV"],
                        "loading": abs(value / rating) * 100,
                        "line_name": p["NAME"],
                    }
                )

    gens, loads = [], []
    totals = {
        f"Pg{fuel}{suffix}": 0 for fuel in (*COLORS, "other") for suffix in ("", "cap")
    }
    for point in points:
        p, coords = point["properties"], point["geometry"]["coordinates"]
        pg, capacity, fuel, count = 0, 0, "other", 0
        for bus in p["bus"]:
            for gen in bus.get("gen", []):
                power, cap = (
                    gen["GEN_STATUS"] * gen["PG"],
                    gen["GEN_STATUS"] * gen["PMAX"],
                )
                pg += power
                capacity += cap
                fuel = (gen.get("GEN_FUEL") or "other").lower()
                category = fuel if fuel in COLORS else "other"
                totals[f"Pg{category}"] += power
                totals[f"Pg{category}cap"] += cap
                count += 1
        if count:
            gens.append(
                {
                    "coordinates": coords,
                    "Pg": pg,
                    "Pcap": capacity,
                    "KVlevels": [bus.get("BASE_KV") for bus in p["bus"]],
                    "color": COLORS.get(fuel, "black"),
                    "fuel": fuel,
                    "name": p["NAME"],
                    "countyname": p["countyname"],
                }
            )
        pd = sum(bus.get("PDloss", 0) for bus in p["bus"])
        if pd > 0:
            loads.append({"coordinates": coords, "Pd": pd})
    gendata = {
        **totals,
        "Gens": gens,
        "minPg": min((g["Pg"] for g in gens), default=0),
        "maxPg": max([0] + [g["Pg"] for g in gens]),
    }
    bounds = collection_bounds(grid)
    return {
        "grid_data": grid,
        "areas": hulls(data, "areas"),
        "zones": hulls(data, "zones"),
        "grid_flowdata": {"locations": locations, "flows": flows, "maxloading": 120},
        "grid_flowdata_reactive": {
            "locations": locations,
            "flows": reactive,
            "maxloading": 120,
        },
        "Points": [
            {
                "coordinates": p["geometry"]["coordinates"],
                "value": p["properties"]["bus"][0]["VM"],
            }
            for p in points
        ],
        "Vcontour": [
            {"threshold": limits, "color": color}
            for limits, color in (
                ([0.9, 0.98], [255, 0, 0]),
                ([0.98, 1.02], [0, 255, 0]),
                ([1.02, 1.1], [0, 0, 255]),
            )
        ],
        "gendata": gendata,
        "generation": gens,
        "loaddata": {
            "Loads": loads,
            "minPd": min([1000] + [p["Pd"] for p in loads]),
            "maxPd": max([0] + [p["Pd"] for p in loads]),
        },
        "countymaxPd": max_pd,
        "countyload": countyload,
        "countyloaddata": {"maxPd": max_pd},
        "bounds": bounds,
        "mapcenter": {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Point",
                "coordinates": [
                    (bounds[0] + bounds[2]) / 2,
                    (bounds[1] + bounds[3]) / 2,
                ],
            },
        },
    }


def collection_bounds(grid):
    bounds = [shape(f["geometry"]).bounds for f in grid["features"]]
    return [
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    ]
