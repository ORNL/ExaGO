export function filterFlows(data, {
  voltageRange = [0, Infinity], loadingRange = [0, Infinity],
  lineNames = [], busNames = [],
} = {}) {
  const flows = data.flows.filter((flow) =>
    voltageRange[0] <= flow.count && flow.count <= voltageRange[1]
    && loadingRange[0] <= flow.loading && flow.loading <= loadingRange[1]
    && (!lineNames.length || lineNames.includes(flow.line_name))
    && (lineNames.length || !busNames.length || busNames.includes(flow.origin) || busNames.includes(flow.dest))
  ).map(({ origin, dest, count, loading }) => ({ origin, dest, count, loading }));
  const unique = [...new Map(flows.map((flow) => [JSON.stringify(flow), flow])).values()];
  return { ...data, flows: unique };
}
