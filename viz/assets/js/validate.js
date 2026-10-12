export function validateNetworkData(data) {
  const features = data?.geojsondata?.features;
  if (data?.geojsondata?.type !== "FeatureCollection" || !Array.isArray(features) || !features.length) {
    throw new Error("Choose an ExaGO JSON output containing geojsondata and network features.");
  }
  for (const feature of features) {
    const { geometry, properties } = feature;
    if (!properties || typeof properties.NAME !== "string") {
      throw new Error("Every network feature must have a name and properties.");
    }
    if (geometry?.type !== "Point" && geometry?.type !== "LineString") {
      throw new Error("Network features must be points or transmission lines.");
    }
    const coordinates = geometry.type === "Point" ? [geometry.coordinates] : geometry.coordinates;
    if (!Array.isArray(coordinates) || !coordinates.length || coordinates.some((p) =>
      !Array.isArray(p) || !Number.isFinite(p[0]) || !Number.isFinite(p[1]) || Math.abs(p[1]) >= 90)) {
      throw new Error("Network coordinates must contain valid longitude and latitude values.");
    }
    if (geometry.type === "Point" && (
      !Array.isArray(properties.KVlevels) || !Array.isArray(properties.bus) || !properties.bus.length ||
      properties.bus.some((bus) => !Number.isFinite(bus.VM)))) {
      throw new Error("Each substation must contain bus voltages and voltage levels.");
    }
  }
  return data;
}
