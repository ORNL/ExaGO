import { convex, bbox, booleanPointInPolygon } from "@turf/turf";
import us from "us";

const countydata = await fetch("/assets/data/counties.json").then(r => r.json());
const codeDict = us.mapping("fips", "abbr");

function getCountyNodes(data) {
  const points = data.features.filter((f) => f.geometry.type === "Point");
  for (const point of points) {
    const buses = point.properties.bus;
    point.properties.Vm = buses.reduce((sum, bus) => sum + bus.VM, 0) / buses.length;
    point.properties.Va = buses[0].VA;
    point.properties.Pd = buses.reduce((sum, bus) => sum + (bus.PD ?? 0), 0);
    point.properties.Qd = buses.reduce((sum, bus) => sum + (bus.QD ?? 0), 0);
    point.properties.bus_type = buses[0].BUS_TYPE;
    point.properties.countyname = "";
  }
  const counties = [];
  let maxPd = 0;
  for (const polygon of countydata.features) {
    const box = bbox(polygon);
    const properties = { ...polygon.properties, Pd: 0, Vm_avg: 0, KVlevels: [] };
    properties.countyname = `${properties.NAME}, ${codeDict[properties.STATE] ?? properties.STATE}`;
    let busCount = 0;
    for (const point of points) {
      const [lng, lat] = point.geometry.coordinates;
      if (box[0] <= lng && lng <= box[2] && box[1] <= lat && lat <= box[3]
          && booleanPointInPolygon(point, polygon)) {
        point.properties.countyname = properties.countyname;
        properties.KVlevels.push(...point.properties.KVlevels);
        for (const bus of point.properties.bus) {
          properties.Pd += bus.PDloss ?? 0;
          properties.Vm_avg += bus.VM;
          busCount++;
        }
      }
    }
    if (busCount) {
      properties.Vm_avg /= busCount;
      maxPd = Math.max(maxPd, properties.Pd);
      counties.push({ ...polygon, properties });
    }
  }
  return { maxPd, data: { type: "FeatureCollection", features: counties }, updatedata: data };
}

// Extract data for first time-slice
function ExtractFirstTimeSlice(data) {
  var features = data.features.filter(function (feature) {
    if (!("start" in feature.properties) || feature.properties.start == 0) return feature;
  });
  var gdata = { type: "FeatureCollection", features: features };
  return gdata;
}

function ExtractFlowData(data, {
  voltageRange = [0, Infinity], activeRange = [0, Infinity], reactiveRange = [0, Infinity],
  lineNames = [], busNames = [],
} = {}) {
  const locations = data.features.filter((f) => f.geometry.type === "Point").map((f) => ({
    id: f.properties.NAME, name: f.properties.NAME,
    lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1],
  }));
  const locationIds = new Set(locations.map((location) => location.id));
  const flows = [], reactiveFlows = [];
  for (const feature of data.features) {
    if (feature.geometry.type !== "LineString") continue;
    const p = feature.properties;
    const [a, b] = p.NAME.split(" -- ");
    if (!locationIds.has(a) || !locationIds.has(b)) continue;
    if (p.KV < voltageRange[0] || p.KV > voltageRange[1]) continue;
    if (lineNames.length && !lineNames.includes(p.NAME)) continue;
    if (!lineNames.length && busNames.length && !busNames.includes(a) && !busNames.includes(b)) continue;
    const rating = p.RATE_A > 0 ? p.RATE_A : 10000;
    const loading = Math.abs(p.PF / rating) * 100;
    const reactiveLoading = Math.abs(p.QF / rating) * 100;
    if (Number.isFinite(loading) && activeRange[0] <= loading && loading <= activeRange[1]) {
      flows.push({ origin: p.PF >= 0 ? a : b, dest: p.PF >= 0 ? b : a, count: p.KV, loading });
    }
    if (Number.isFinite(reactiveLoading) && reactiveRange[0] <= reactiveLoading && reactiveLoading <= reactiveRange[1]) {
      reactiveFlows.push({ origin: p.QF >= 0 ? a : b, dest: p.QF >= 0 ? b : a, count: p.KV, loading: reactiveLoading });
    }
  }
  const unique = (items) => [...new Map(items.map((item) => [JSON.stringify(item), item])).values()];
  return [
    { locations, flows: unique(flows), maxloading: 120 },
    { locations, flows: unique(reactiveFlows), maxloading: 120 },
  ];
}

// get net data for bar
function getBarNet(data) {
  const netBarValue = data.features.map((line) => line.properties.KV);
  return netBarValue;
}

// Get substation voltages
function getPoints(data) {
  var Pointsi;
  var Points = [];
  var elev;
  var i,
    k = 0;
  for (i = 0; i < data.features.length; i++) {
    if (data.features[i].geometry.type == "Point") {
      Pointsi = { coordinates: data.features[i].geometry.coordinates, value: data.features[i].properties.bus[0].VM };
      Points.push(Pointsi);
    }
  }
  return Points;
}

// Get generation
function getGeneration(data) {
  var Geni;
  var Gens = [];
  var minPg = Infinity;
  var maxPg = 0.0;
  var elev;
  var i, j, k;
  var Pgcoal = 0.0,
    Pghydro = 0.0,
    Pgnuclear = 0.0,
    Pgng = 0.0,
    Pgsolar = 0.0,
    Pgwind = 0.0,
    Pgother = 0.0;
  var Pgcoalcap = 0.0,
    Pghydrocap = 0.0,
    Pgnuclearcap = 0.0,
    Pgngcap = 0.0,
    Pgsolarcap = 0.0,
    Pgwindcap = 0.0,
    Pgothercap = 0.0;
  for (i = 0; i < data.features.length; i++) {
    if (data.features[i].geometry.type == "Point") {
      var subst = data.features[i].properties;
      var nbus = subst.bus.length;
      var Pg = 0.0;
      var Pcap = 0.0;
      var gen_fuel;
      var ngen = 0;
      var KV = [];
      var name = subst.NAME;
      var countyname = subst.countyname;
      for (j = 0; j < nbus; j++) {
        var bus = subst.bus[j];
        KV.push(bus.BASE_KV);
        for (k = 0; k < (bus.gen?.length ?? 0); k++) {
          var gen = bus.gen[k];
          Pg += gen.GEN_STATUS * gen.PG;
          Pcap += gen.GEN_STATUS * gen.PMAX;
          gen_fuel = (gen.GEN_FUEL ?? "other").toLowerCase();

          if (gen_fuel == "wind") {
            Pgwind += gen.GEN_STATUS * gen.PG;
            Pgwindcap += gen.GEN_STATUS * gen.PMAX;
          } else if (gen_fuel == "solar") {
            Pgsolar += gen.GEN_STATUS * gen.PG;
            Pgsolarcap += gen.GEN_STATUS * gen.PMAX;
          } else if (gen_fuel == "coal") {
            Pgcoal += gen.GEN_STATUS * gen.PG;
            Pgcoalcap += gen.GEN_STATUS * gen.PMAX;
          } else if (gen_fuel == "nuclear") {
            Pgnuclear += gen.GEN_STATUS * gen.PG;
            Pgnuclearcap += gen.GEN_STATUS * gen.PMAX;
          } else if (gen_fuel == "hydro") {
            Pghydro += gen.GEN_STATUS * gen.PG;
            Pghydrocap += gen.GEN_STATUS * gen.PMAX;
          } else if (gen_fuel == "ng") {
            Pgng += gen.GEN_STATUS * gen.PG;
            Pgngcap += gen.GEN_STATUS * gen.PMAX;
          } else {
            Pgother += gen.GEN_STATUS * gen.PG;
            Pgothercap += gen.GEN_STATUS * gen.PMAX;
          }
          ngen++;
        }
      }
      if (ngen) {
        var color;
        if (gen_fuel == "wind") color = "green";
        else if (gen_fuel == "solar") color = "yellow";
        else if (gen_fuel == "coal") color = "gray";
        else if (gen_fuel == "nuclear") color = "red";
        else if (gen_fuel == "hydro") color = "blue";
        else if (gen_fuel == "ng") color = "orange";
        else color = "black";
        if (Pg <= minPg) minPg = Pg;
        if (Pg >= maxPg) maxPg = Pg;
        Geni = { coordinates: data.features[i].geometry.coordinates, Pg: Pg, Pcap: Pcap, KVlevels: KV, color: color, fuel: gen_fuel, name: name, countyname: countyname };
        Gens.push(Geni);
      }
    }
  }

  if (!Gens.length) minPg = 0;
  return { minPg: minPg, maxPg: maxPg, Gens: Gens, Pgwind: Pgwind, Pgsolar: Pgsolar, Pgnuclear: Pgnuclear, Pghydro: Pghydro, Pgng: Pgng, Pgcoal: Pgcoal, Pgother: Pgother, Pgwindcap: Pgwindcap, Pgsolarcap: Pgsolarcap, Pgnuclearcap: Pgnuclearcap, Pghydrocap: Pghydrocap, Pgngcap: Pgngcap, Pgcoalcap: Pgcoalcap, Pgothercap: Pgothercap };
}

// Get Load
function getLoad(data) {
  var Loadi;
  var Loads = [];
  var minPd = 1000.0;
  var maxPd = 0.0;
  var elev;
  var i, j;
  for (i = 0; i < data.features.length; i++) {
    if (data.features[i].geometry.type == "Point") {
      var subst = data.features[i].properties;
      var nbus = subst.bus.length;
      var Pd = 0.0;
      var Qd = 0.0;
      for (j = 0; j < nbus; j++) {
        var bus = subst.bus[j];
        Pd += bus.PDloss;
        Qd += bus.QDloss;
      }
      if (Pd > 0) {
        if (Pd <= minPd) minPd = Pd;
        if (Pd >= maxPd) maxPd = Pd;
        Loadi = { coordinates: data.features[i].geometry.coordinates, Pd: Pd };
        Loads.push(Loadi);
      }
    }
  }
  return { minPd: minPd, maxPd: maxPd, Loads: Loads };
}

function getContours() {
  var Vmin = 0.9;
  var Vmax = 1.1;

  var contours = [
    { threshold: [0.9, 0.98], color: [255, 0, 0] },
    { threshold: [0.98, 1.02], color: [0, 255, 0] },
    { threshold: [1.02, 1.1], color: [0, 0, 255] },
  ];
  return contours;
}

// Filter features with geometry type "Point" and properties area = area_num
function getArea(data, area_num) {
  var filteredFeatures = data.features.filter((feature) => feature.geometry.type === "Point" && feature.properties.area === area_num);

  // Create a new GeoJSON object with filtered features
  var filteredGeoJSON = {
    type: "FeatureCollection",
    features: filteredFeatures,
  };

  var areahull = convex(filteredGeoJSON);
  if (areahull) {
    areahull.properties = { name: area_num };
  }

  return areahull;
}

function getAreas(sysdata) {
  var areas = [];

  for (var i = 0; i < (sysdata.areas?.length ?? 0); i++) {
    var area = getArea(sysdata.geojsondata, sysdata.areas[i]);
    if (area) {
      areas.push(area);
    }
  }

  return { type: "FeatureCollection", features: areas };
}

// Filter features with geometry type "Point" and properties zone = zone_num
function getZone(data, zone_num) {
  var filteredFeatures = data.features.filter((feature) => feature.geometry.type === "Point" && feature.properties.zone === zone_num);

  // Create a new GeoJSON object with filtered features
  var filteredGeoJSON = {
    type: "FeatureCollection",
    features: filteredFeatures,
  };

  var zonehull = convex(filteredGeoJSON);
  if (zonehull) {
    zonehull.properties = { name: zone_num };
  }

  return zonehull;
}

function getZones(sysdata) {
  var zones = [];

  for (var i = 0; i < (sysdata.zones?.length ?? 0); i++) {
    var zone = getZone(sysdata.geojsondata, sysdata.zones[i]);
    if (zone) {
      zones.push(zone);
    }
  }

  return { type: "FeatureCollection", features: zones };
}

export { getCountyNodes, ExtractFirstTimeSlice, ExtractFlowData, getBarNet, getPoints, getGeneration, getLoad, getContours, getAreas, getZones };
