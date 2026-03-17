import { StrictMode, useMemo, useRef, useState, useCallback, useEffect } from "react";

// DATA
import {
  getCountyNodes,
  ExtractFirstTimeSlice,
  ExtractFlowData,
  getPoints,
  getGeneration,
  getLoad,
  getContours,
  getAreas,
  getZones,
} from "./dataprocess.js";

import { FlowmapLayer, PickingType } from "@flowmap.gl/layers";

import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";

// MUI
import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormGroup from "@mui/material/FormGroup";
import { Typography } from "@mui/material";
import HomeOutlinedIcon from "@mui/icons-material/HomeOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import Slider from "@mui/material/Slider";
import Box from "@mui/material/Box";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Divider from "@mui/material/Divider";
import { Multiselect } from "react-widgets";

import { Chart as ChartJS, RadialLinearScale, ArcElement, Tooltip, Legend } from "chart.js";
import { Doughnut } from "react-chartjs-2";

import { DataFilterExtension } from "@deck.gl/extensions";
import { LinearInterpolator, FlyToInterpolator } from "deck.gl";

import { center, convex, bbox } from "@turf/turf";

import ChatBot from "react-chatbotify";

import { Map, NavigationControl, FullscreenControl, useControl } from "react-map-gl/maplibre";
import { WebMercatorViewport } from "@deck.gl/core";

import { GeoJsonLayer, ColumnLayer } from "@deck.gl/layers";

import {
  LineColor,
  FlowColor,
  FillColor,
  fillGenColumnColor,
  fillGenColumnColorCap,
  getVoltageFillColor,
} from "./color.js";

ChartJS.register(RadialLinearScale, ArcElement, Tooltip, Legend);

// Transition interpolators
const transitionLinearInterpolator = new LinearInterpolator(["bearing"]);
const transitionFlyToInterpolator = new FlyToInterpolator(["zoom"]);

const MAP_STYLE = {
  pos_no_label: "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json",
  pos: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  none: "",
};

import "maplibre-gl/dist/maplibre-gl.css";

// ─── Persistent settings ──────────────────────────────────────────────────────
const SETTINGS_KEY = "exago-grid-settings-v2";

const DEFAULT_SETTINGS = {
  // ── Map appearance ──────────────────────────────────────────────────────
  mapStyle: "pos",
  // ── Active power flow ───────────────────────────────────────────────────
  activeFlowAnimate: false,
  activeFlowClustering: true,
  activeFlowAdaptiveScales: true,
  activeFlowOpacity: 0.8,
  activeFlowColorStart: "#0000ff",
  activeFlowColorEnd: "#ff00ff",
  // ── Reactive power flow ─────────────────────────────────────────────────
  reactiveFlowAnimate: false,
  reactiveFlowClustering: true,
  reactiveFlowAdaptiveScales: true,
  reactiveFlowOpacity: 0.8,
  reactiveFlowColorStart: "#00c8ff",
  reactiveFlowColorEnd: "#ff9500",
  // ── Network display ─────────────────────────────────────────────────────
  showCountyBoundaries: true,
  showStateBoundaries: true,
  countyBoundaryColor: "#c8c8c8",
  stateBoundaryColor: "#646464",
  lineWidthScale: 0.005,
  networkPointRadius: 3,
  // ── Generation bars ─────────────────────────────────────────────────────
  genBarRadius: 15000,
  genBarAutoRadius: true,   // scale bar footprint with zoom level
  genElevationScale: 100,      // metres per MW  (Pg × scale)
};

/**
 * Like useState but reads/writes to localStorage under `key`.
 * On first load merges stored value with defaults so new keys survive upgrades.
 */
function useLocalStorage(key, defaultValue) {
  const [value, setValue] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored ? { ...defaultValue, ...JSON.parse(stored) } : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      console.warn("localStorage write failed");
    }
  }, [key, value]);

  // Convenience: patch a single key without replacing the whole object
  const patch = useCallback(
    (partial) => setValue((prev) => ({ ...prev, ...partial })),
    []
  );

  return [value, setValue, patch];
}

// ─── SettingsDialog ───────────────────────────────────────────────────────────

/** Convert #rrggbb → [r, g, b] for deck.gl */
function hexToRgb(hex) {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Convert #rrggbb → "rgb(r,g,b)" string for FlowmapLayer colorScheme */
function hexToRgbStr(hex) {
  const [r, g, b] = hexToRgb(hex);
  return `rgb(${r},${g},${b})`;
}

function SettingsDialog({ open, onClose, settings, onPatch }) {
  const [draft, setDraft] = useState(settings);
  useEffect(() => { if (open) setDraft(settings); }, [open]);

  const patch = (partial) => setDraft((d) => ({ ...d, ...partial }));
  const handleSave = () => { onPatch(draft); onClose(); };
  const handleReset = () => setDraft(DEFAULT_SETTINGS);

  const SwitchRow = ({ label, field }) => (
    <FormControlLabel
      control={
        <Switch
          checked={draft[field]}
          onChange={(e) => patch({ [field]: e.target.checked })}
          size="small"
        />
      }
      label={<span style={{ fontSize: 13 }}>{label}</span>}
      style={{ marginBottom: 2 }}
    />
  );

  const ColorRow = ({ label, field }) => (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
      <input
        type="color"
        value={draft[field]}
        onChange={(e) => patch({ [field]: e.target.value })}
        style={{ width: 32, height: 26, padding: 1, border: "1px solid #ccc", borderRadius: 4, cursor: "pointer", background: "none" }}
      />
      <span style={{ fontSize: 13, color: "#444" }}>{label}</span>
      <span style={{ marginLeft: "auto", fontSize: 11, color: "#999", fontFamily: "monospace" }}>{draft[field]}</span>
    </div>
  );

  const GradientRow = ({ label, startField, endField }) => (
    <div style={{ marginBottom: 10 }}>
      <span style={{ fontSize: 12, color: "#555", display: "block", marginBottom: 6 }}>{label}</span>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <input
          type="color"
          value={draft[startField]}
          onChange={(e) => patch({ [startField]: e.target.value })}
          style={{ width: 32, height: 26, padding: 1, border: "1px solid #ccc", borderRadius: 4, cursor: "pointer", background: "none" }}
        />
        <div style={{
          flex: 1, height: 14, borderRadius: 7,
          background: `linear-gradient(to right, ${draft[startField]}, ${draft[endField]})`,
          border: "1px solid #ddd",
        }} />
        <input
          type="color"
          value={draft[endField]}
          onChange={(e) => patch({ [endField]: e.target.value })}
          style={{ width: 32, height: 26, padding: 1, border: "1px solid #ccc", borderRadius: 4, cursor: "pointer", background: "none" }}
        />
      </div>
    </div>
  );

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth
      PaperProps={{ style: { borderRadius: 10 } }}>
      <DialogTitle style={{ paddingBottom: 8, fontWeight: 700, fontSize: 16 }}>
        ⚙️ Settings
      </DialogTitle>

      <DialogContent dividers style={{ display: "flex", flexDirection: "column", gap: 16 }}>

        {/* ── Map style ───────────────────────────────────────────── */}
        <div>
          <div style={sectionLabel}>Map Style</div>
          <FormControl size="small" fullWidth>
            <Select value={draft.mapStyle} onChange={(e) => patch({ mapStyle: e.target.value })}>
              <MenuItem value="pos">Positron (default)</MenuItem>
              <MenuItem value="pos_no_label">Positron – no labels</MenuItem>
              <MenuItem value="dark">Dark Matter</MenuItem>
              <MenuItem value="none">No basemap</MenuItem>
            </Select>
          </FormControl>
        </div>

        <Divider />

        {/* ── Active power flow ───────────────────────────────────── */}
        <div>
          <div style={sectionLabel}>Active Power Flow</div>
          <FormGroup>
            <SwitchRow label="Enable animation" field="activeFlowAnimate" />
            <SwitchRow label="Enable clustering" field="activeFlowClustering" />
            <SwitchRow label="Adaptive scales" field="activeFlowAdaptiveScales" />
          </FormGroup>
          <div style={{ marginTop: 8 }}>
            <GradientRow
              label="Colour gradient  (low loading → high loading)"
              startField="activeFlowColorStart"
              endField="activeFlowColorEnd"
            />
            <span style={{ fontSize: 12, color: "#555" }}>
              Opacity: <strong>{Math.round(draft.activeFlowOpacity * 100)}%</strong>
            </span>
            <Slider
              value={draft.activeFlowOpacity}
              min={0.05} max={1} step={0.05}
              onChange={(_, v) => patch({ activeFlowOpacity: v })}
              size="small"
            />
          </div>
        </div>

        <Divider />

        {/* ── Reactive power flow ─────────────────────────────────── */}
        <div>
          <div style={sectionLabel}>Reactive Power Flow</div>
          <FormGroup>
            <SwitchRow label="Enable animation" field="reactiveFlowAnimate" />
            <SwitchRow label="Enable clustering" field="reactiveFlowClustering" />
            <SwitchRow label="Adaptive scales" field="reactiveFlowAdaptiveScales" />
          </FormGroup>
          <div style={{ marginTop: 8 }}>
            <GradientRow
              label="Colour gradient  (low loading → high loading)"
              startField="reactiveFlowColorStart"
              endField="reactiveFlowColorEnd"
            />
            <span style={{ fontSize: 12, color: "#555" }}>
              Opacity: <strong>{Math.round(draft.reactiveFlowOpacity * 100)}%</strong>
            </span>
            <Slider
              value={draft.reactiveFlowOpacity}
              min={0.05} max={1} step={0.05}
              onChange={(_, v) => patch({ reactiveFlowOpacity: v })}
              size="small"
            />
          </div>
        </div>

        <Divider />

        {/* ── Network display ─────────────────────────────────────── */}
        <div>
          <div style={sectionLabel}>Network Display</div>

          <div style={{ marginBottom: 10 }}>
            <span style={{ fontSize: 12, color: "#555" }}>
              Node circle radius: <strong>{draft.networkPointRadius} px</strong>
            </span>
            <Slider
              value={draft.networkPointRadius}
              min={2} max={20} step={1}
              onChange={(_, v) => patch({ networkPointRadius: v })}
              size="small"
            />
          </div>

          <div style={{ marginBottom: 10 }}>
            <span style={{ fontSize: 12, color: "#555" }}>
              Line width scale: <strong>{draft.lineWidthScale.toFixed(3)}</strong>
            </span>
            <Slider
              value={draft.lineWidthScale}
              min={0.001} max={0.05} step={0.001}
              onChange={(_, v) => patch({ lineWidthScale: v })}
              size="small"
            />
          </div>

          <FormGroup style={{ marginBottom: 6 }}>
            <SwitchRow label="Show county boundaries" field="showCountyBoundaries" />
          </FormGroup>
          {draft.showCountyBoundaries && (
            <div style={{ paddingLeft: 8, marginBottom: 8 }}>
              <ColorRow label="County boundary colour" field="countyBoundaryColor" />
            </div>
          )}

          <FormGroup style={{ marginBottom: 6 }}>
            <SwitchRow label="Show state boundaries" field="showStateBoundaries" />
          </FormGroup>
          {draft.showStateBoundaries && (
            <div style={{ paddingLeft: 8 }}>
              <ColorRow label="State boundary colour" field="stateBoundaryColor" />
            </div>
          )}
        </div>

        <Divider />

        {/* ── Generation bars ─────────────────────────────────────── */}
        <div>
          <div style={sectionLabel}>Generation Bars</div>

          <div style={{ marginBottom: 10 }}>
            <span style={{ fontSize: 12, color: "#555" }}>
              Bar width (radius): <strong>{(draft.genBarRadius / 1000).toFixed(1)} km</strong>
            </span>
            <Slider
              value={draft.genBarRadius}
              min={500} max={50000} step={500}
              onChange={(_, v) => patch({ genBarRadius: v })}
              size="small"
            />
          </div>

          <FormGroup style={{ marginBottom: 4 }}>
            <SwitchRow label="Auto-scale radius with zoom" field="genBarAutoRadius" />
          </FormGroup>
          <span style={{ fontSize: 11, color: "#aaa", display: "block", marginBottom: 10 }}>
            When on, bar width shrinks as you zoom in so bars stay proportional on screen.
            The radius above sets the size at zoom&nbsp;4.
          </span>

          <div style={{ marginBottom: 4 }}>
            <span style={{ fontSize: 12, color: "#555" }}>
              Height scale: <strong>{draft.genElevationScale}×</strong>
            </span>
            <Slider
              value={draft.genElevationScale}
              min={1} max={100} step={1}
              onChange={(_, v) => patch({ genElevationScale: v })}
              size="small"
            />
          </div>
          <span style={{ fontSize: 11, color: "#aaa", display: "block", marginTop: 2 }}>
            Bar height = Pg × scale
          </span>
        </div>

      </DialogContent>

      <DialogActions style={{ padding: "10px 16px", gap: 8 }}>
        <Button size="small" onClick={handleReset} color="inherit">Restore defaults</Button>
        <Button size="small" onClick={onClose} color="inherit">Cancel</Button>
        <Button size="small" onClick={handleSave} variant="contained" disableElevation>Save</Button>
      </DialogActions>
    </Dialog>
  );
}

const sectionLabel = {
  fontSize: 11,
  fontWeight: 700,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  color: "#888",
  marginBottom: 8,
};


// ─── DeckOverlay ─────────────────────────────────────────────────────────────
function DeckOverlay({ layers, onClick }) {
  const overlay = useControl(
    () =>
      new MapboxOverlay({
        interleaved: true,
        layers,
        onClick,
      })
  );

  useEffect(() => {
    overlay.setProps({ layers, onClick });
  }, [overlay, layers, onClick]);

  return null;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function valuetext(value) {
  return `${value.toFixed(2)}`;
}

function rgba([r, g, b], a = 1) {
  return `rgba(${r},${g},${b},${a})`;
}

function LineWidth(line) {
  return 300;
  //   return line.properties.KV * 3;
  //return Math.abs(line.properties.PF/line.properties.RATE_A)*500;
}

// ─── Legend ──────────────────────────────────────────────────────────────────
const KV_BINS = [
  { max: 39.4, color: [151, 220, 248], label: "<= 39 kV" },
  { max: 68, color: [103, 205, 244], label: "40 - 67 kV" },
  { max: 161, color: [107, 172, 197], label: "68 - 161 kV" },
  { max: 230, color: [145, 35, 255], label: "162 – 230 kV" },
  { max: 350, color: [255, 0, 255], label: "231 - 350 kV" },
  { max: 500, color: [255, 150, 11], label: "351 – 500 kV" },
  { max: 900, color: [231, 102, 34], label: ">= 501 kV" },
];

function ColorLegend({ title = "Voltage (kV)", bins = KV_BINS }) {
  return (
    <div style={legendWrap}>
      <div style={legendTitle}>{title}</div>
      {bins.map((b, i) => (
        <div key={i} style={row}>
          <span style={{ ...swatch, background: rgba(b.color) }} />
          <span style={label}>{b.label}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Constants ───────────────────────────────────────────────────────────────
const INITIAL_VIEW_STATE = {
  latitude: 39.8283,
  longitude: -98.5795,
  zoom: 4,
  maxZoom: 16,
  pitch: 0,
  bearing: 0,
};

// For now it's only the default file. TODO: need to get rid of relative path.
const DATA_FILES = Object.keys(import.meta.glob("../data/opflowout.json", { eager: true })).map((path) =>
  path.split("/").pop()
);

// ─── App ─────────────────────────────────────────────────────────────────────
function App({
  refdata,
  refflowdata,
  refflowdata_reactive,
  ggdata,
  gendata,
  generation,
  areas,
  zones,
  countyload,
  countyloaddata,
  countymaxPd,
  mapcenter,
  homeViewState,
  selected,
  onSelectCase,
  onFileUpload,
  mapStyle = MAP_STYLE,
}) {
  const mapRef = useRef(null);

  // ── Persistent settings ─────────────────────────────────────────────────
  const [settings, , patchSettings] = useLocalStorage(SETTINGS_KEY, DEFAULT_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [gridData, setGridData] = useState(refdata);

  // ── Selection widget state ──────────────────────────────────────────────
  const [nameSelectItems, setNameSelectItems] = useState([]);
  const [nameItems] = useState(gendata.Gens.map((g) => g.name));

  const [lineNameSelectItems, setLineNameSelectItems] = useState([]);
  const [lineNameItems] = useState(
    gridData.features.filter((f) => f.geometry.type === "LineString").map((f) => f.properties.NAME)
  );

  const [busNameSelectItems, setBusNameSelectItems] = useState([]);
  const [busNameItems] = useState(
    gridData.features.filter((f) => f.geometry.type === "Point").map((f) => f.properties.NAME)
  );

  const [areaNameSelectItems, setAreaNameSelectItems] = useState([]);
  const [areaNameItems] = useState(areas.features.map((f) => f.properties.name));

  const [zoneNameSelectItems, setZoneNameSelectItems] = useState([]);
  const [zoneNameItems] = useState(zones.features.map((f) => f.properties.name));

  const [countyNameSelectItems, setCountyNameSelectItems] = useState([]);
  const [countyNameItems] = useState(
    countyload.features.map((c) => c.properties.countyname)
  );

  // ── Flow data ───────────────────────────────────────────────────────────
  const [flowdata, setFlowData] = useState(refflowdata);
  const [reactiveflowdata, setReactivFlowData] = useState(refflowdata_reactive);

  // ── Filter sliders ──────────────────────────────────────────────────────
  const [genfiltervalue, setGenFilterValue] = useState([gendata.minPg, gendata.maxPg]);
  const [netfiltervalue, setNetFilterValue] = useState([0, 800]);
  const [flowfiltervalue, setFlowFilterValue] = useState([0, 120]);
  const [flowfilterreactivevalue, setFlowFilterReactiveValue] = useState([0, 120]);
  const [loadfiltervalue, setLoadFilterValue] = useState([0, countyloaddata.maxPd]);
  const [voltagefiltervalue, setVoltageFilterValue] = useState([0.89, 1.11]);

  // ── View / UI state — seed from the extent-fitted homeViewState ─────────
  const [initialViewState, setInitialViewState] = useState(homeViewState);
  const [showPopup, setShowPopup] = useState({ display: false, info: "", name: "" });
  const [tooltip, setTooltip] = useState();

  // Fly to the new dataset extent whenever the source data changes (dropdown
  // switch or new file upload both produce a fresh homeViewState object).
  // Preserve the current pitch so the camera angle isn't reset on load.
  useEffect(() => {
    setInitialViewState((vs) => ({
      ...homeViewState,
      pitch: vs.pitch,          // keep whatever pitch the user currently has
      transitionInterpolator: transitionFlyToInterpolator,
      transitionDuration: 1800,
    }));
    setShowPopup({ display: false, info: "", name: "" });
  }, [homeViewState]);

  // ── Generation chart ────────────────────────────────────────────────────
  const [genDoughlabels, setDoughlabels] = useState([
    "Wind", "Solar", "Nuclear", "Natural Gas", "Hydro", "Coal", "Other",
  ]);
  const colorMap = {
    green: "Wind", yellow: "Solar", gray: "Coal",
    red: "Nuclear", blue: "Hydro", orange: "Natural Gas", black: "Other",
  };

  // ── Layer visibility ────────────────────────────────────────────────────
  const [netlayeractive, setNetLayerActive] = useState(true);
  const [flowlayeractive, setFlowLayerActive] = useState(true);
  const [reactiveflowlayeractive, setReactiveFlowLayerActive] = useState(false);
  const [loadlayeractive, setLoadLayerActive] = useState(false);
  const [genlayeractive, setGenLayerActive] = useState(false);
  const [genlayercapactive, setGenLayerCapActive] = useState(false);
  const [voltagelayeractive, setVoltageLayerActive] = useState(false);
  const [zonelayeractive, setZoneLayerActive] = useState(false);
  const [arealayeractive, setAreaLayerActive] = useState(false);

  // ── Screenshot state ────────────────────────────────────────────────────
  const [screenshotUrl, setScreenshotUrl] = useState(null);

  // ─── Rebuild flow data whenever filters or selections change ─────────────
  useEffect(() => {
    setGridData(refdata);
    setFlowData(refflowdata);
    setReactivFlowData(refflowdata_reactive);

    const locations = [];
    const flows = [];
    const reactive_flows = [];

    if (lineNameSelectItems.length > 0) {
      const pointNames = lineNameSelectItems.map((n) => n.split(" -- ")).flat();
      gridData.features.forEach((feature) => {
        if (feature.geometry.type === "Point" && pointNames.includes(feature.properties.NAME)) {
          locations.push({
            id: feature.properties.NAME,
            name: feature.properties.NAME,
            lon: feature.geometry.coordinates[0],
            lat: feature.geometry.coordinates[1],
          });
        } else if (
          feature.geometry.type === "LineString" &&
          lineNameSelectItems.includes(feature.properties.NAME)
        ) {
          const RATE_A = feature.properties.RATE_A === 0 ? 10000 : feature.properties.RATE_A;
          const loading = Math.abs(feature.properties.PF / RATE_A) * 100;
          const [a, b] = feature.properties.NAME.split(" -- ");
          flows.push({
            origin: feature.properties.PF > 0 ? a : b,
            dest: feature.properties.PF > 0 ? b : a,
            count: feature.properties.KV,
            loading,
          });
        }
      });
    } else if (busNameSelectItems.length > 0) {
      gridData.features.forEach((feature) => {
        if (feature.geometry.type === "Point" && busNameSelectItems.includes(feature.properties.NAME)) {
          locations.push({
            id: feature.properties.NAME,
            name: feature.properties.NAME,
            lon: feature.geometry.coordinates[0],
            lat: feature.geometry.coordinates[1],
          });
        } else if (
          feature.geometry.type === "LineString" &&
          feature.properties.NAME.split(" -- ").some((r) => busNameSelectItems.includes(r))
        ) {
          const RATE_A = feature.properties.RATE_A === 0 ? 10000 : feature.properties.RATE_A;
          const loading = Math.abs(feature.properties.PF / RATE_A) * 100;
          const [a, b] = feature.properties.NAME.split(" -- ");
          flows.push({
            origin: feature.properties.PF > 0 ? a : b,
            dest: feature.properties.PF > 0 ? b : a,
            count: feature.properties.KV,
            loading,
          });
        }
      });
    } else {
      gridData.features.forEach((feature) => {
        if (
          feature.geometry.type === "Point" &&
          netfiltervalue[0] <= feature.properties.KVlevels[0] &&
          feature.properties.KVlevels[0] <= netfiltervalue[1]
        ) {
          locations.push({
            id: feature.properties.NAME,
            name: feature.properties.NAME,
            lon: feature.geometry.coordinates[0],
            lat: feature.geometry.coordinates[1],
          });
        } else if (
          feature.geometry.type === "LineString" &&
          netfiltervalue[0] <= feature.properties.KV &&
          feature.properties.KV <= netfiltervalue[1]
        ) {
          const RATE_A = feature.properties.RATE_A === 0 ? 10000 : feature.properties.RATE_A;
          const loading = Math.abs(feature.properties.PF / RATE_A) * 100;
          const var_loading = Math.abs(feature.properties.QF / RATE_A) * 100;
          const [a, b] = feature.properties.NAME.split(" -- ");

          if (flowfiltervalue[0] <= loading && loading <= flowfiltervalue[1]) {
            flows.push({
              origin: feature.properties.PF > 0 ? a : b,
              dest: feature.properties.PF > 0 ? b : a,
              count: feature.properties.KV,
              loading,
            });
          }

          if (flowfilterreactivevalue[0] <= var_loading && var_loading <= flowfilterreactivevalue[1]) {
            reactive_flows.push({
              origin: feature.properties.QF > 0 ? a : b,
              dest: feature.properties.QF > 0 ? b : a,
              count: feature.properties.KV,
              loading: var_loading,
            });
          }
        }
      });
    }

    setFlowData({ locations, flows, maxloading: 120 });
    setReactivFlowData({ locations, flows: reactive_flows, maxloading: 120 });
  }, [refdata, gridData, netfiltervalue, flowfiltervalue, flowfilterreactivevalue, lineNameSelectItems]);

  // ─── Rotate camera ────────────────────────────────────────────────────────
  // FIX: use a ref instead of a plain var so the value survives re-renders
  const rotatestateRef = useRef(false);

  const rotateCamera = useCallback(() => {
    rotatestateRef.current = !rotatestateRef.current;
    if (rotatestateRef.current) {
      setInitialViewState((vs) => ({
        ...vs,
        bearing: vs.bearing - 180,
        transitionDuration: 20000,
        transitionInterpolator: transitionLinearInterpolator,
        onTransitionEnd: rotateCamera,
      }));
    } else {
      setInitialViewState((vs) => ({ ...vs, onTransitionEnd: null }));
    }
  }, []);

  // ─── View helpers ─────────────────────────────────────────────────────────
  const activatePopup = useCallback(() => {
    setShowPopup((p) => ({ ...p, display: true }));
  }, []);

  const zoomToGen = useCallback((lat, long) => {
    setInitialViewState((vs) => ({
      ...vs,
      latitude: lat,
      longitude: long,
      pitch: 50,
      transitionInterpolator: transitionFlyToInterpolator,
      transitionDuration: 2000,
      zoom: 7.5,
      onTransitionEnd: activatePopup,
    }));
  }, [activatePopup]);

  const zoomToData = useCallback((info) => {
    const lat = info.coordinate[1];
    const long = info.coordinate[0];

    setInitialViewState((vs) => ({
      ...vs,
      latitude: lat,
      longitude: long,
      pitch: 50,
      transitionInterpolator: transitionFlyToInterpolator,
      transitionDuration: 2000,
      zoom: 7.5,
      onTransitionEnd: activatePopup,
    }));

    if (info.layer.id === "geojson") {
      const p = info.object.properties;
      if (info.object.geometry.type === "Point") {
        const kvList = Array.isArray(p.KVlevels) ? p.KVlevels.join(", ") : p.KVlevels;
        const lines = [
          `Voltage level: ${kvList != null ? `${kvList} kV` : "—"}`,
          `|V| (Vm):      ${p.Vm != null ? `${p.Vm.toFixed(4)} p.u.` : "—"}`,
          p.Va != null ? `∠V (Va):       ${p.Va.toFixed(3)}°` : null,
          p.Pd != null ? `Load (Pd):     ${p.Pd.toFixed(2)} MW` : null,
          p.Qd != null ? `Load (Qd):     ${p.Qd.toFixed(2)} MVAR` : null,
          p.bus_type != null ? `Bus type:      ${p.bus_type}` : null,
          p.area != null ? `Area:          ${p.area}` : null,
          p.zone != null ? `Zone:          ${p.zone}` : null,
        ].filter(Boolean);
        setShowPopup((p2) => ({ ...p2, name: p.NAME, info: lines.join("\n") }));
      } else {
        const RATE_A = p.RATE_A > 0 ? p.RATE_A : 10000;
        const pLoading = (Math.abs(p.PF / RATE_A) * 100).toFixed(1);
        const lines = [
          `Voltage:    ${p.KV != null ? `${p.KV} kV` : "—"}`,
          `P from (PF):${p.PF != null ? ` ${p.PF.toFixed(2)} MW` : " —"}`,
          p.PT != null ? `P to (PT):  ${p.PT.toFixed(2)} MW` : null,
          `Q from (QF):${p.QF != null ? ` ${p.QF.toFixed(2)} MVAR` : " —"}`,
          p.QT != null ? `Q to (QT):  ${p.QT.toFixed(2)} MVAR` : null,
          `Rating:     ${p.RATE_A != null ? `${p.RATE_A} MW` : "—"}`,
          `P loading:  ${pLoading}%`,
        ].filter(Boolean);
        setShowPopup((p2) => ({ ...p2, name: p.NAME, info: lines.join("\n") }));
      }
    } else if (info.layer.id === "gen-column") {
      const d = info.object;
      const loading = d.Pcap > 0 ? ((d.Pg / d.Pcap) * 100).toFixed(1) : "—";
      setShowPopup((p) => ({
        ...p,
        name: d.name ?? "Generator",
        info: [
          d.type ? `Type: ${d.type}` : null,
          d.fuel ? `Fuel: ${d.fuel}` : null,
          `Output:   ${d.Pg != null ? d.Pg.toFixed(2) : "—"} MW`,
          `Capacity: ${d.Pcap != null ? d.Pcap.toFixed(2) : "—"} MW`,
          `Loading:  ${loading}%`,
          d.Qg != null ? `Reactive: ${d.Qg.toFixed(2)} MVAR` : null,
          d.Vg != null ? `Voltage:  ${d.Vg.toFixed(4)} p.u.` : null,
        ].filter(Boolean).join("\n"),
      }));
    }
  }, [activatePopup]);

  const zoomToBounds = useCallback(
    (minLng, minLat, maxLng, maxLat, popup = null) => {
      const { longitude, latitude, zoom } = new WebMercatorViewport({
        width: Math.max(window.innerWidth - 260, 400),
        height: Math.max(window.innerHeight, 400),
      }).fitBounds([[minLng, minLat], [maxLng, maxLat]], { padding: 60 });
      setInitialViewState((vs) => ({
        ...vs,
        latitude,
        longitude,
        zoom: Math.min(zoom, 14),
        pitch: 50,
        transitionInterpolator: transitionFlyToInterpolator,
        transitionDuration: 5000,
        onTransitionEnd: activatePopup,
      }));
      if (popup) setShowPopup((p) => ({ ...p, ...popup }));
    },
    [activatePopup]
  );

  const zoomToCountyName = useCallback(
    (minLng, minLat, maxLng, maxLat) => zoomToBounds(minLng, minLat, maxLng, maxLat),
    [zoomToBounds]
  );

  const zoomToAreaName = useCallback(
    (filterarea, minLng, minLat, maxLng, maxLat) =>
      zoomToBounds(minLng, minLat, maxLng, maxLat, {
        display: false,
        name: `Area ${filterarea.properties.name}`,
        info: "",
      }),
    [zoomToBounds]
  );

  const zoomToZoneName = useCallback(
    (filterzone, minLng, minLat, maxLng, maxLat) =>
      zoomToBounds(minLng, minLat, maxLng, maxLat, {
        display: false,
        name: `Zone ${filterzone.properties.name}`,
        info: "",
      }),
    [zoomToBounds]
  );

  const zoomToCounty = useCallback((info) => {
    if (!info || !["PolygonLayer2", "PolygonLayerload"].includes(info.layer?.id)) return;
    const [c1, c2, c3, c4] = bbox(info.object);
    const { longitude, latitude, zoom } = info.layer.context.viewport.fitBounds([[c1, c2], [c3, c4]]);
    setInitialViewState((vs) => ({
      ...vs,
      latitude,
      longitude,
      pitch: 50,
      transitionInterpolator: transitionFlyToInterpolator,
      transitionDuration: 5000,
      zoom: zoom - 0.25,
      onTransitionEnd: activatePopup,
    }));
    setShowPopup({ display: false, name: info.object.properties.NAME, info: `Load loss: ${info.object.properties.Pd.toFixed(2)}MW` });
  }, [activatePopup]);

  const zoomToArea = useCallback((info) => {
    if (!info || info.layer?.id !== "AreaLayer") return;
    const [c1, c2, c3, c4] = bbox(info.object);
    const { longitude, latitude, zoom } = info.layer.context.viewport.fitBounds([[c1, c2], [c3, c4]]);
    setInitialViewState((vs) => ({
      ...vs,
      latitude,
      longitude,
      pitch: 50,
      transitionInterpolator: transitionFlyToInterpolator,
      transitionDuration: 5000,
      zoom: zoom - 0.25,
      onTransitionEnd: activatePopup,
    }));
    setShowPopup({ display: false, name: `Area ${info.object.properties.name}`, info: "" });
  }, [activatePopup]);

  const zoomToZone = useCallback((info) => {
    if (!info || info.layer?.id !== "ZoneLayer") return;
    const [c1, c2, c3, c4] = bbox(info.object);
    const { longitude, latitude, zoom } = info.layer.context.viewport.fitBounds([[c1, c2], [c3, c4]]);
    setInitialViewState((vs) => ({
      ...vs,
      latitude,
      longitude,
      pitch: 50,
      transitionInterpolator: transitionFlyToInterpolator,
      transitionDuration: 5000,
      zoom: zoom - 0.25,
      onTransitionEnd: activatePopup,
    }));
    setShowPopup({ display: false, name: `Zone ${info.object.properties.name}`, info: "" });
  }, [activatePopup]);

  const GoHome = useCallback(() => {
    // Fly back to the full extent of the current dataset
    setInitialViewState({
      ...homeViewState,
      transitionInterpolator: transitionFlyToInterpolator,
      transitionDuration: 1200,
    });
    setShowPopup((p) => ({ ...p, display: false }));
  }, [homeViewState]);

  // ─── Screenshot ───────────────────────────────────────────────────────────
  // MapboxOverlay with interleaved:true shares MapLibre's single WebGL context,
  // so mapRef.current.getCanvas() captures basemap + all deck.gl layers in one shot.
  // preserveDrawingBuffer={true} must be set on <Map> (see below).
  const takeScreenshot = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const canvas = map.getCanvas();
    const url = canvas.toDataURL("image/png");
    setScreenshotUrl(url);
  }, []);

  const downloadScreenshot = useCallback(() => {
    if (!screenshotUrl) return;
    const a = document.createElement("a");
    a.href = screenshotUrl;
    a.download = `grid-screenshot-${Date.now()}.png`;
    a.click();
  }, [screenshotUrl]);

  // ─── Filter value accessors ───────────────────────────────────────────────
  function getNetFilterValue(data) {
    if (!data) return 10000;
    if (lineNameSelectItems.length > 0) {
      const pointNames = lineNameSelectItems.map((n) => n.split(" -- ")).flat();
      if (data.geometry.type === "Point" && pointNames.includes(data.properties.NAME))
        return data.properties.KVlevels[0];
      if (data.geometry.type === "LineString" && lineNameSelectItems.includes(data.properties.NAME))
        return data.properties.KV;
    } else if (busNameSelectItems.length > 0) {
      if (data.geometry.type === "Point" && busNameSelectItems.includes(data.properties.NAME))
        return data.properties.KVlevels[0];
      if (
        data.geometry.type === "LineString" &&
        data.properties.NAME.split(" -- ").some((r) => busNameSelectItems.includes(r))
      )
        return data.properties.KV;
    } else {
      if (data.geometry.type === "Point") {
        for (const KV of data.properties.KVlevels) {
          if (netfiltervalue[0] <= KV && KV <= netfiltervalue[1]) return KV;
        }
      } else if (data.geometry.type === "LineString") {
        return data.properties.KV;
      }
    }
    return -1;
  }

  function getGenFilterValue(data) {
    if (!data) return 10000;
    if (genDoughlabels.length > 0 && !genDoughlabels.includes(colorMap[data.color])) return 10000;
    const inKV = data.KVlevels.some((KV) => netfiltervalue[0] <= KV && KV <= netfiltervalue[1]);
    if (nameSelectItems.length > 0)
      return nameSelectItems.includes(data.name) && inKV ? data.Pg : 10000;
    return inKV ? data.Pg : 10000;
  }

  function getLoadFilterValue(data) {
    if (!data) return -10000;
    const inKV = data.properties.KVlevels.some((KV) => netfiltervalue[0] <= KV && KV <= netfiltervalue[1]);
    if (countyNameSelectItems.length > 0)
      return countyNameSelectItems.includes(data.properties.countyname) && inKV
        ? data.properties.Pd
        : -10000;
    return inKV ? data.properties.Pd : -10000;
  }

  function getVoltageFilterValue(data) {
    if (!data) return -10000;
    const inKV = data.properties.KVlevels.some((KV) => netfiltervalue[0] <= KV && KV <= netfiltervalue[1]);
    if (countyNameSelectItems.length > 0)
      return countyNameSelectItems.includes(data.properties.countyname) && inKV
        ? data.properties.Vm_avg
        : -10000;
    return inKV ? data.properties.Vm_avg : -10000;
  }

  // ─── ChatBot ──────────────────────────────────────────────────────────────
  async function fetchMyData(params) {
    try {
      const response = await fetch("http://localhost:5000/data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputText: params.userInput }),
      });
      const chatOutput = await response.json();
      const chatList = chatOutput.result_list;

      if (chatList.length > 0) {
        const keyList = Object.keys(chatList[0]);
        const containsCap = keyList.some((s) => s.includes("capacity"));

        if ("generation_name" in chatList[0]) {
          const names = chatList.map((d) => d.generation_name);
          setGenLayerActive(true);
          if (containsCap) setGenLayerCapActive(true);
          setNameSelectItems(names);
          setGenFilterValue([gendata.minPg, gendata.maxPg]);
          setInitialViewState((vs) => ({ ...vs, pitch: 40, transitionInterpolator: transitionFlyToInterpolator, transitionDuration: 2000 }));
        }
        if ("line_name" in chatList[0]) {
          setNetLayerActive(true);
          setFlowLayerActive(true);
          setBusNameSelectItems([]);
          setLineNameSelectItems(chatList.map((d) => d.line_name));
        }
        if ("bus_name" in chatList[0]) {
          setNetLayerActive(true);
          setFlowLayerActive(true);
          setLineNameSelectItems([]);
          setBusNameSelectItems(chatList.map((d) => d.bus_name));
        }
      }
      return chatOutput.text;
    } catch {
      return "Sorry, I didn't find an answer to your question. Please try rephrasing.";
    }
  }

  const flow = {
    start: {
      message:
        "Hello! What do you want to know about the power grid data? You can ask me to show specific generation units, transmission lines, or buses by name.",
      path: "loop",
    },
    loop: {
      message: async (params) => fetchMyData(params),
      path: "loop",
    },
  };

  // ─── Event handlers ───────────────────────────────────────────────────────
  const handleNetLayerChange = (e) => { setNetLayerActive(e.target.checked); setNetFilterValue([0, 800]); };
  const handleFlowLayerChange = (e) => { setFlowLayerActive(e.target.checked); setFlowFilterValue([0, 120]); };
  const handleReactiveFlowLayerChange = (e) => { setReactiveFlowLayerActive(e.target.checked); setFlowFilterReactiveValue([0, 120]); };

  const handleLoadLayerChange = (e) => {
    setLoadLayerActive(e.target.checked);
    setLoadFilterValue([0, countyloaddata.maxPd]);
    if (e.target.checked)
      setInitialViewState((vs) => ({ ...vs, pitch: 40, transitionInterpolator: transitionFlyToInterpolator, transitionDuration: 2000 }));
  };

  const handleVoltageLayerChange = (e) => {
    setVoltageLayerActive(e.target.checked);
    setVoltageFilterValue([0.89, 1.11]);
    if (e.target.checked)
      setInitialViewState((vs) => ({ ...vs, pitch: 40, transitionInterpolator: transitionFlyToInterpolator, transitionDuration: 2000 }));
  };

  const handleAreaLayerChange = (e) => {
    setAreaLayerActive(e.target.checked);
    if (e.target.checked)
      setInitialViewState((vs) => ({ ...vs, pitch: 40, transitionInterpolator: transitionFlyToInterpolator, transitionDuration: 2000 }));
  };

  const handleZoneLayerChange = (e) => {
    setZoneLayerActive(e.target.checked);
    if (e.target.checked)
      setInitialViewState((vs) => ({ ...vs, pitch: 40, transitionInterpolator: transitionFlyToInterpolator, transitionDuration: 2000 }));
  };

  const handleGenLayerChange = (e) => {
    setGenLayerActive(e.target.checked);
    setGenFilterValue([gendata.minPg, gendata.maxPg]);
    if (!e.target.checked) setGenLayerCapActive(false);
    if (e.target.checked)
      setInitialViewState((vs) => ({ ...vs, pitch: 40, transitionInterpolator: transitionFlyToInterpolator, transitionDuration: 2000 }));
  };

  const handleGenLayerCapChange = (e) => {
    setGenLayerCapActive(e.target.checked);
    setGenFilterValue([gendata.minPg, gendata.maxPg]);
    if (e.target.checked)
      setInitialViewState((vs) => ({ ...vs, pitch: 40, transitionInterpolator: transitionFlyToInterpolator, transitionDuration: 2000 }));
  };

  const handleGenRangeFilterChange = (e) => setGenFilterValue(e.target.value);
  const handleLoadRangeFilterChange = (e) => setLoadFilterValue(e.target.value);
  const handleVoltageRangeFilterChange = (e) => setVoltageFilterValue(e.target.value);
  const handleNetRangeFilterChange = (e) => setNetFilterValue(e.target.value);
  const handleFlowRangeFilterChange = (e) => setFlowFilterValue(e.target.value);
  const handleReactiveFlowRangeFilterChange = (e) => setFlowFilterReactiveValue(e.target.value);

  // ─── Multiselect handlers ─────────────────────────────────────────────────
  const handleBusMultiselect = (_, meta) => {
    setLineNameSelectItems([]);
    setBusNameSelectItems((prev) => {
      const idx = prev.indexOf(meta.dataItem);
      return idx >= 0 ? prev.filter((_, i) => i !== idx) : [...prev, meta.dataItem];
    });
  };

  const handleLineMultiselect = (_, meta) => {
    setBusNameSelectItems([]);
    setLineNameSelectItems((prev) => {
      const idx = prev.indexOf(meta.dataItem);
      return idx >= 0 ? prev.filter((_, i) => i !== idx) : [...prev, meta.dataItem];
    });
  };

  const handleGenMultiselect = (_, meta) => {
    setNameSelectItems((prev) => {
      const idx = prev.indexOf(meta.dataItem);
      if (idx >= 0) return prev.filter((_, i) => i !== idx);
      const gen = gendata.Gens.find((g) => g.name === meta.dataItem);
      if (gen) zoomToGen(gen.coordinates[1], gen.coordinates[0]);
      return [...prev, meta.dataItem];
    });
  };

  const handleCountyMultiselect = (_, meta) => {
    setCountyNameSelectItems((prev) => {
      const idx = prev.indexOf(meta.dataItem);
      if (idx >= 0) return prev.filter((_, i) => i !== idx);
      const county = countyload.features.find((c) => c.properties.countyname === meta.dataItem);
      if (county) {
        const lons = county.geometry.coordinates[0].map((d) => d[0]);
        const lats = county.geometry.coordinates[0].map((d) => d[1]);
        zoomToCountyName(Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats));
      }
      return [...prev, meta.dataItem];
    });
  };

  const handleAreaMultiselect = (_, meta) => {
    setAreaNameSelectItems((prev) => {
      const idx = prev.indexOf(meta.dataItem);
      if (idx >= 0) return prev.filter((_, i) => i !== idx);
      const area = areas.features.find((a) => a.properties.name === meta.dataItem);
      if (area) {
        const lons = area.geometry.coordinates[0].map((d) => d[0]);
        const lats = area.geometry.coordinates[0].map((d) => d[1]);
        zoomToAreaName(area, Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats));
      }
      return [...prev, meta.dataItem];
    });
  };

  const handleZoneMultiselect = (_, meta) => {
    setZoneNameSelectItems((prev) => {
      const idx = prev.indexOf(meta.dataItem);
      if (idx >= 0) return prev.filter((_, i) => i !== idx);
      const zone = zones.features.find((z) => z.properties.name === meta.dataItem);
      if (zone) {
        const lons = zone.geometry.coordinates[0].map((d) => d[0]);
        const lats = zone.geometry.coordinates[0].map((d) => d[1]);
        zoomToZoneName(zone, Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats));
      }
      return [...prev, meta.dataItem];
    });
  };

  const handleDoughnutClick = (_, legendItem, legend) => {
    if (legendItem.hidden) {
      setDoughlabels((prev) => [...prev, legendItem.text]);
    } else {
      setDoughlabels((prev) => prev.filter((l) => l !== legendItem.text));
    }
    legend.chart.toggleDataVisibility(legendItem.index);
  };

  // ─── Layers ───────────────────────────────────────────────────────────────
  const filterExt = [new DataFilterExtension({ filtersize: 1 })];

  // Zoom-adaptive bar radius: at the reference zoom (4) radius == genBarRadius.
  // Each zoom step halves/doubles the geographic size of one screen pixel, so
  // multiplying by 2^(refZoom - currentZoom) keeps bars visually constant-sized.
  const RADIUS_REF_ZOOM = 4;
  const effectiveGenRadius = settings.genBarAutoRadius
    ? settings.genBarRadius * Math.pow(2, RADIUS_REF_ZOOM - (initialViewState.zoom ?? RADIUS_REF_ZOOM))
    : settings.genBarRadius;

  const layer_flow_active = new FlowmapLayer({
    id: "layer-flow",
    data: flowdata,
    visible: flowlayeractive,
    animationEnabled: settings.activeFlowAnimate,
    colorScheme: [
      hexToRgbStr(settings.activeFlowColorStart),
      hexToRgbStr(settings.activeFlowColorEnd),
    ],
    opacity: settings.activeFlowOpacity,
    clusteringEnabled: settings.activeFlowClustering,
    adaptiveScalesEnabled: settings.activeFlowAdaptiveScales,
    getFlowMagnitude: (f) => f.loading,
    getFlowOriginId: (f) => f.origin,
    getFlowDestId: (f) => f.dest,
    getLocationId: (l) => l.id,
    getLocationLat: (l) => l.lat,
    getLocationLon: (l) => l.lon,
    getLocationName: (l) => l.name,
    pickable: true,
    onHover: (info) => setTooltip(getTooltipState(info)),
    onClick: (info) => console.log("clicked", info.object?.type, info.object),
  });

  const layer_flow_reactive = new FlowmapLayer({
    id: "layer-flow-reactive",
    data: reactiveflowdata,
    visible: reactiveflowlayeractive,
    animationEnabled: settings.reactiveFlowAnimate,
    colorScheme: [
      hexToRgbStr(settings.reactiveFlowColorStart),
      hexToRgbStr(settings.reactiveFlowColorEnd),
    ],
    opacity: settings.reactiveFlowOpacity,
    clusteringEnabled: settings.reactiveFlowClustering,
    adaptiveScalesEnabled: settings.reactiveFlowAdaptiveScales,
    getFlowMagnitude: (f) => f.loading,
    getFlowOriginId: (f) => f.origin,
    getFlowDestId: (f) => f.dest,
    getLocationId: (l) => l.id,
    getLocationLat: (l) => l.lat,
    getLocationLon: (l) => l.lon,
    getLocationName: (l) => l.name,
    pickable: true,
    onHover: (info) => setTooltip(getTooltipState(info)),
    onClick: (info) => console.log("clicked", info.object?.type, info.object),
  });

  const layer_network = new GeoJsonLayer({
    id: "geojson",
    data: gridData,
    stroked: false,
    filled: true,
    extruded: true,
    pickable: netlayeractive,
    visible: netlayeractive,
    pointType: "circle",
    lineWidthScale: settings.lineWidthScale,
    lineWidthUnits: "pixels",
    getFillColor: FillColor,
    getLineColor: LineColor,
    getPointRadius: settings.networkPointRadius,
    pointRadiusUnits: "pixels",
    getLineWidth: LineWidth,
    onHover: (info) => {
      if (!info?.object) return setTooltip(undefined);
      const p = info.object.properties;
      const isLine = info.object.geometry.type === "LineString";

      let content;
      if (isLine) {
        const RATE_A = p.RATE_A > 0 ? p.RATE_A : 10000;
        const pLoading = Math.abs(p.PF / RATE_A) * 100;
        const qLoading = Math.abs(p.QF / RATE_A) * 100;
        content = (
          <div style={{ minWidth: 210, fontFamily: "Inter, system-ui, sans-serif" }}>
            <div style={ttHead}>{p.NAME}</div>
            <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
              <tbody>
                <tr><td style={ttLabel}>Voltage</td>      <td style={ttVal}>{p.KV != null ? `${p.KV} kV` : "—"}</td></tr>
                <tr><td style={ttLabel}>P from (PF)</td>  <td style={ttVal}>{p.PF != null ? `${p.PF.toFixed(2)} MW` : "—"}</td></tr>
                {p.PT != null && <tr><td style={ttLabel}>P to (PT)</td>    <td style={ttVal}>{p.PT.toFixed(2)} MW</td></tr>}
                <tr><td style={ttLabel}>Q from (QF)</td>  <td style={ttVal}>{p.QF != null ? `${p.QF.toFixed(2)} MVAR` : "—"}</td></tr>
                {p.QT != null && <tr><td style={ttLabel}>Q to (QT)</td>    <td style={ttVal}>{p.QT.toFixed(2)} MVAR</td></tr>}
                <tr><td style={ttLabel}>Rating (A)</td>   <td style={ttVal}>{p.RATE_A != null ? `${p.RATE_A} MW` : "—"}</td></tr>
                <tr><td style={ttLabel}>P loading</td>    <td style={{ ...ttVal, color: pLoading > 90 ? "#ff6b6b" : pLoading > 70 ? "#ffd93d" : "#6bffb8" }}>{pLoading.toFixed(1)}%</td></tr>
                {p.QF != null && p.RATE_A > 0 && <tr><td style={ttLabel}>Q loading</td><td style={ttVal}>{qLoading.toFixed(1)}%</td></tr>}
              </tbody>
            </table>
          </div>
        );
      } else {
        const kvList = Array.isArray(p.KVlevels) ? p.KVlevels.join(", ") : p.KVlevels;
        content = (
          <div style={{ minWidth: 200, fontFamily: "Inter, system-ui, sans-serif" }}>
            <div style={ttHead}>{p.NAME}</div>
            <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
              <tbody>
                <tr><td style={ttLabel}>Voltage level</td> <td style={ttVal}>{kvList != null ? `${kvList} kV` : "—"}</td></tr>
                <tr><td style={ttLabel}>|V| (Vm)</td>       <td style={ttVal}>{p.Vm != null ? `${p.Vm.toFixed(4)} p.u.` : "—"}</td></tr>
                {p.Va != null && <tr><td style={ttLabel}>∠V (Va)</td>       <td style={ttVal}>{p.Va.toFixed(3)}°</td></tr>}
                {p.Pd != null && <tr><td style={ttLabel}>Load (Pd)</td>     <td style={ttVal}>{p.Pd.toFixed(2)} MW</td></tr>}
                {p.Qd != null && <tr><td style={ttLabel}>Load (Qd)</td>     <td style={ttVal}>{p.Qd.toFixed(2)} MVAR</td></tr>}
                {p.bus_type != null && <tr><td style={ttLabel}>Bus type</td>  <td style={ttVal}>{p.bus_type}</td></tr>}
                {p.area != null && <tr><td style={ttLabel}>Area</td>          <td style={ttVal}>{p.area}</td></tr>}
                {p.zone != null && <tr><td style={ttLabel}>Zone</td>          <td style={ttVal}>{p.zone}</td></tr>}
              </tbody>
            </table>
          </div>
        );
      }

      setTooltip({
        position: { left: info.x + 12, top: info.y - 10 },
        content,
        hoveredCountyId: info.object.id,
      });
    },
    onClick: zoomToData,
    getFilterValue: getNetFilterValue,
    filterRange: netfiltervalue,
    extensions: filterExt,
    updateTriggers: {
      getFilterValue: [netfiltervalue, lineNameSelectItems, busNameSelectItems, flowfiltervalue, flowfilterreactivevalue],
    },
  });

  const layer_generator_power = new ColumnLayer({
    id: "gen-column",
    data: generation,
    diskResolution: 50,
    radius: effectiveGenRadius,
    elevationScale: 1,
    pickable: genlayeractive,
    visible: genlayeractive,
    getPosition: (d) => d.coordinates,
    getFillColor: fillGenColumnColor,
    getElevation: (d) => d.Pg * settings.genElevationScale,
    onHover: (info) => {
      if (!info?.object) return setTooltip(undefined);
      const d = info.object;
      const loading = d.Pcap > 0 ? ((d.Pg / d.Pcap) * 100).toFixed(1) : "—";
      setTooltip({
        position: { left: info.x + 12, top: info.y - 10 },
        content: (
          <div style={{ minWidth: 200, fontFamily: "Inter, system-ui, sans-serif" }}>
            <div style={ttHead}>
              {d.name ?? "Generator"}
            </div>
            <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
              <tbody>
                {d.type != null && <tr><td style={ttLabel}>Type</td>        <td style={ttVal}>{d.type}</td></tr>}
                {d.fuel != null && <tr><td style={ttLabel}>Fuel</td>        <td style={ttVal}>{d.fuel}</td></tr>}
                {d.bus != null && <tr><td style={ttLabel}>Bus</td>         <td style={ttVal}>{d.bus}</td></tr>}
                {d.area != null && <tr><td style={ttLabel}>Area</td>        <td style={ttVal}>{d.area}</td></tr>}
                {d.zone != null && <tr><td style={ttLabel}>Zone</td>        <td style={ttVal}>{d.zone}</td></tr>}
                <tr><td style={ttLabel}>Output (Pg)</td>   <td style={ttVal}><strong>{d.Pg != null ? d.Pg.toFixed(2) : "—"} MW</strong></td></tr>
                <tr><td style={ttLabel}>Capacity (Pcap)</td><td style={ttVal}><strong>{d.Pcap != null ? d.Pcap.toFixed(2) : "—"} MW</strong></td></tr>
                <tr><td style={ttLabel}>Loading</td>       <td style={ttVal}>{loading}%</td></tr>
                {d.Qg != null && <tr><td style={ttLabel}>Reactive (Qg)</td><td style={ttVal}>{d.Qg.toFixed(2)} MVAR</td></tr>}
                {d.Vg != null && <tr><td style={ttLabel}>Voltage (Vg)</td> <td style={ttVal}>{d.Vg.toFixed(4)} p.u.</td></tr>}
                {d.status != null && <tr><td style={ttLabel}>Status</td>       <td style={ttVal}>{d.status}</td></tr>}
              </tbody>
            </table>
          </div>
        ),
      });
    },
    // onClick: zoomToData,
    getFilterValue: getGenFilterValue,
    filterRange: genfiltervalue,
    extensions: filterExt,
    updateTriggers: {
      getFilterValue: [netfiltervalue, genDoughlabels, nameSelectItems],
      getElevation: [settings.genElevationScale],
    },
  });

  const layer_generator_capacity = new ColumnLayer({
    id: "gen-column-cap",
    data: generation,
    diskResolution: 50,
    radius: effectiveGenRadius,
    elevationScale: 1,
    pickable: true,
    visible: genlayercapactive,
    getPosition: (d) => d.coordinates,
    getFillColor: fillGenColumnColorCap,
    getElevation: (d) => d.Pcap * settings.genElevationScale,
    onHover: (info) => {
      if (!info?.object) return setTooltip(undefined);
      const d = info.object;
      const loading = d.Pcap > 0 ? ((d.Pg / d.Pcap) * 100).toFixed(1) : "—";
      setTooltip({
        position: { left: info.x + 12, top: info.y - 10 },
        content: (
          <div style={{ minWidth: 200, fontFamily: "Inter, system-ui, sans-serif" }}>
            <div style={ttHead}>
              {d.name ?? "Generator"}
            </div>
            <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
              <tbody>
                {d.type != null && <tr><td style={ttLabel}>Type</td>        <td style={ttVal}>{d.type}</td></tr>}
                {d.fuel != null && <tr><td style={ttLabel}>Fuel</td>        <td style={ttVal}>{d.fuel}</td></tr>}
                {d.bus != null && <tr><td style={ttLabel}>Bus</td>         <td style={ttVal}>{d.bus}</td></tr>}
                {d.area != null && <tr><td style={ttLabel}>Area</td>        <td style={ttVal}>{d.area}</td></tr>}
                {d.zone != null && <tr><td style={ttLabel}>Zone</td>        <td style={ttVal}>{d.zone}</td></tr>}
                <tr><td style={ttLabel}>Output (Pg)</td>   <td style={ttVal}><strong>{d.Pg != null ? d.Pg.toFixed(2) : "—"} MW</strong></td></tr>
                <tr><td style={ttLabel}>Capacity (Pcap)</td><td style={ttVal}><strong>{d.Pcap != null ? d.Pcap.toFixed(2) : "—"} MW</strong></td></tr>
                <tr><td style={ttLabel}>Loading</td>       <td style={ttVal}>{loading}%</td></tr>
                {d.Qg != null && <tr><td style={ttLabel}>Reactive (Qg)</td><td style={ttVal}>{d.Qg.toFixed(2)} MVAR</td></tr>}
                {d.Vg != null && <tr><td style={ttLabel}>Voltage (Vg)</td> <td style={ttVal}>{d.Vg.toFixed(4)} p.u.</td></tr>}
                {d.status != null && <tr><td style={ttLabel}>Status</td>       <td style={ttVal}>{d.status}</td></tr>}
              </tbody>
            </table>
          </div>
        ),
      });
    },
    getFilterValue: getGenFilterValue,
    filterRange: genfiltervalue,
    extensions: filterExt,
    updateTriggers: {
      getFilterValue: [netfiltervalue, genDoughlabels, nameSelectItems],
      getElevation: [settings.genElevationScale],
    },
  });

  const layer_area = new GeoJsonLayer({
    id: "AreaLayer",
    data: areas,
    pickable: arealayeractive,
    visible: arealayeractive,
    stroked: true,
    filled: true,
    extruded: true,
    wireframe: true,
    lineWidthMinPixels: 1,
    getFillColor: [255, 192, 203],
    getLineColor: [80, 80, 80],
    getLineWidth: 1,
    opacity: 0.1,
    onClick: zoomToArea,
  });

  const layer_zone = new GeoJsonLayer({
    id: "ZoneLayer",
    data: zones,
    pickable: zonelayeractive,
    visible: zonelayeractive,
    stroked: true,
    filled: true,
    extruded: true,
    wireframe: true,
    lineWidthMinPixels: 1,
    getFillColor: [252, 245, 95],
    getLineColor: [80, 80, 80],
    getLineWidth: 1,
    opacity: 0.1,
    onClick: zoomToZone,
  });

  const layer_county_load = new GeoJsonLayer({
    id: "PolygonLayerload",
    data: countyload,
    pickable: loadlayeractive,
    visible: loadlayeractive,
    stroked: true,
    filled: true,
    extruded: true,
    wireframe: true,
    lineWidthMinPixels: 1,
    getFillColor: (d) => [(255 * d.properties.Pd) / countymaxPd, 0, 0],
    getLineColor: [80, 80, 80],
    getLineWidth: 1,
    opacity: 0.1,
    onClick: zoomToCounty,
    extensions: filterExt,
    getFilterValue: getLoadFilterValue,
    filterRange: loadfiltervalue,
    updateTriggers: { getFilterValue: [netfiltervalue, countyNameSelectItems] },
  });

  const layer_county_voltage = new GeoJsonLayer({
    id: "PolygonLayer2",
    data: countyload,
    pickable: voltagelayeractive,
    visible: voltagelayeractive,
    stroked: true,
    filled: true,
    extruded: true,
    wireframe: true,
    lineWidthMinPixels: 1,
    getFillColor: getVoltageFillColor,
    getLineColor: [80, 80, 80],
    getLineWidth: 1,
    opacity: 0.1,
    onClick: zoomToCounty,
    extensions: filterExt,
    getFilterValue: getVoltageFilterValue,
    filterRange: voltagefiltervalue,
    updateTriggers: { getFilterValue: [netfiltervalue, countyNameSelectItems] },
  });

  const layer_county_id = new GeoJsonLayer({
    id: "layer-1",
    data: "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    pickable: true,
    stroked: true,
    filled: true,
    visible: settings.showCountyBoundaries,
    getFillColor: (f) =>
      tooltip?.hoveredCountyId && (f.id || f.properties?.GEOID) === tooltip.hoveredCountyId
        ? [255, 230, 120, 200]
        : [255, 250, 200, 10],
    getLineColor: hexToRgb(settings.countyBoundaryColor),
    getLineWidth: 1,
    lineWidthUnits: "pixels",
    onHover: (info) => {
      if (!info?.object) return setTooltip(undefined);
      const p = info.object.properties || {};
      const name = p.NAME || p.name || p.COUNTY || p.county || info.object.id;
      setTooltip({
        position: { left: info.x, top: info.y },
        content: <div>{name}</div>,
        hoveredCountyId: info.object.id,
      });
    },
  });

  const layer_states_id = new GeoJsonLayer({
    id: "layer-2",
    data: "/geo_data/gz_2010_us_040_00_500k.json",
    pickable: false,
    stroked: true,
    filled: true,
    visible: settings.showStateBoundaries,
    getFillColor: [255, 255, 255, 0],
    getLineColor: hexToRgb(settings.stateBoundaryColor),
    getLineWidth: 2,
    lineWidthUnits: "pixels",
  });

  const layers = [
    layer_county_id,
    layer_states_id,
    layer_zone,
    layer_area,
    layer_county_load,
    layer_county_voltage,
    layer_network,
    layer_flow_active,
    layer_flow_reactive,
    layer_generator_capacity,
    layer_generator_power,
  ];

  // ─── Generation chart data ────────────────────────────────────────────────
  const genmixlabels = ["Wind", "Solar", "Nuclear", "Natural Gas", "Hydro", "Coal", "Other"];
  const genmix = [gendata.Pgwind, gendata.Pgsolar, gendata.Pgnuclear, gendata.Pgng, gendata.Pghydro, gendata.Pgcoal, gendata.Pgother];
  const genmixcap = [gendata.Pgwindcap, gendata.Pgsolarcap, gendata.Pgnuclearcap, gendata.Pgngcap, gendata.Pghydrocap, gendata.Pgcoalcap, gendata.Pgothercap];

  const chartdata = {
    labels: genmixlabels,
    datasets: [
      {
        label: "Generation Mix Cap",
        data: genmixcap,
        backgroundColor: [
          "rgba(0,255,0,0.3)", "rgba(244,219,135,0.3)", "rgba(255,0,0,0.3)",
          "rgba(255,165,0,0.3)", "rgba(28,163,236,0.3)", "rgba(128,128,128,0.3)", "rgba(0,0,0,0.3)",
        ],
        borderWidth: 1,
      },
      {
        label: "Generation Mix",
        data: genmix,
        backgroundColor: ["rgb(0,255,0)", "rgb(244,219,135)", "red", "orange", "rgb(28,163,236)", "gray", "black"],
        borderWidth: 1,
      },
    ],
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div>
      <div style={{ width: "100vw", height: "100vh" }}>
        {/*
          preserveDrawingBuffer={true} is required so toDataURL() works on the
          WebGL canvas. Without it the buffer is cleared after each frame.
        */}
        <Map
          {...initialViewState}
          onMove={(evt) => setInitialViewState(evt.viewState)}
          ref={mapRef}
          mapLib={maplibregl}
          mapStyle={MAP_STYLE[settings.mapStyle]}
          reuseMaps
          preventStyleDiffing
          preserveDrawingBuffer={true}
        >
          <NavigationControl position="top-left" />
          <FullscreenControl position="top-left" />

          <div
            style={{
              position: "absolute", top: 150, left: 10,
              width: 30, background: "#fff", color: "#6b6b76", zIndex: 999999,
            }}
          >
            <HomeOutlinedIcon fontSize="medium" onClick={GoHome} style={{ cursor: "pointer" }} />
          </div>

          <DeckOverlay
            layers={layers}
            onClick={(info) => info?.object && console.log("Clicked:", info.object)}
          />
        </Map>
      </div>

      {/* ── Control panel ─────────────────────────────────────────────────── */}
      <div
        style={{
          position: "absolute", top: 0, right: 0, width: 300,
          background: "#fff", padding: "10px 12px", color: "#6b6b76", zIndex: 1000,
          maxHeight: "100vh", overflowY: "auto", overflowX: "hidden", boxSizing: "border-box",
        }}
      >

        {/* ── File upload ─────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "#888", marginBottom: 4 }}>Upload JSON</div>
          <input
            type="file"
            accept=".json"
            style={{ width: "100%", fontSize: 12, boxSizing: "border-box" }}
            onChange={(e) => {
              const file = e.target.files[0];
              if (file) onFileUpload(file);
            }}
          />
        </div>

        {/* ── Screenshot ──────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "#888", marginBottom: 4 }}>Screenshot</div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button onClick={takeScreenshot} style={btnStyle("#3b82f6")}>📷 Capture</button>
            {screenshotUrl && <button onClick={downloadScreenshot} style={btnStyle("#10b981")}>⬇ Download</button>}
            {screenshotUrl && <button onClick={() => setScreenshotUrl(null)} style={btnStyle("#6b7280")}>✕ Clear</button>}
          </div>
          {screenshotUrl && (
            <img
              src={screenshotUrl}
              alt="Screenshot preview"
              style={{ marginTop: 8, width: "100%", borderRadius: 6, border: "1px solid #ddd", boxShadow: "0 2px 8px rgba(0,0,0,0.15)" }}
            />
          )}
        </div>

        <hr style={{ margin: "8px 0", border: "none", borderTop: "1px solid #e5e5e5" }} />

        {/* Header row: Reset + Settings gear */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
          <button
            style={{
              flex: 1, minWidth: 0, padding: "5px 8px",
              backgroundColor: "#f0f0f0", border: "1px solid #ccc",
              borderRadius: 4, cursor: "pointer", fontSize: 11,
              whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
            }}
            onClick={() => {
              setNetFilterValue([0, 800]);
              setFlowFilterValue([0, 120]);
              setFlowFilterReactiveValue([0, 120]);
              setGenFilterValue([gendata.minPg, gendata.maxPg]);
              setLoadFilterValue([0, countyloaddata.maxPd]);
              setVoltageFilterValue([0.89, 1.11]);
              setNameSelectItems([]);
              setLineNameSelectItems([]);
              setBusNameSelectItems([]);
              setCountyNameSelectItems([]);
              setAreaNameSelectItems([]);
              setZoneNameSelectItems([]);
              setDoughlabels(["Wind", "Solar", "Nuclear", "Natural Gas", "Hydro", "Coal", "Other"]);
            }}
          >
            Reset Filters
          </button>
          <button
            title="Settings"
            onClick={() => setSettingsOpen(true)}
            style={{
              flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              width: 32, height: 30,
              background: "#f0f0f0", border: "1px solid #ccc",
              borderRadius: 4, cursor: "pointer", color: "#444", padding: 0,
            }}
          >
            <SettingsOutlinedIcon style={{ fontSize: 18 }} />
          </button>
        </div>

        <div>

          {/* Network */}
          <Accordion defaultExpanded>
            <AccordionSummary style={accordionSummaryStyle} expandIcon={<ArrowDropDownIcon />}>
              <Typography>
                <Checkbox checked={netlayeractive} onChange={handleNetLayerChange} />
                Net
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography component="div">
                <ColorLegend />
                {netlayeractive && (
                  <>
                    <div style={{ paddingRight: 40 }}>
                      <Slider value={netfiltervalue} valueLabelDisplay="auto" onChange={handleNetRangeFilterChange} getAriaValueText={valuetext} step={1} min={0} max={800} />
                    </div>
                    <div style={{ paddingRight: 40 }}>
                      <Multiselect defaultValue={busNameSelectItems} data={busNameItems} placeholder="Search for buses" onChange={handleBusMultiselect} />
                    </div>
                  </>
                )}
              </Typography>
            </AccordionDetails>
          </Accordion>

          {/* Active Flow */}
          <Accordion defaultExpanded>
            <AccordionSummary style={accordionSummaryStyle} expandIcon={<ArrowDropDownIcon />}>
              <Typography>
                <Checkbox checked={flowlayeractive} onChange={handleFlowLayerChange} />
                Flow
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography component="div">
                {flowlayeractive && (
                  <>
                    <div style={{ paddingRight: 40 }}>
                      <Slider value={flowfiltervalue} valueLabelDisplay="auto" onChange={handleFlowRangeFilterChange} getAriaValueText={valuetext} step={10} min={0} max={120} />
                    </div>
                    <div style={{ paddingRight: 40 }}>
                      <Multiselect defaultValue={lineNameSelectItems} data={lineNameItems} placeholder="Search for transmission lines" onChange={handleLineMultiselect} />
                    </div>
                  </>
                )}
              </Typography>
            </AccordionDetails>
          </Accordion>

          {/* Reactive Flow */}
          <Accordion defaultExpanded>
            <AccordionSummary style={accordionSummaryStyle} expandIcon={<ArrowDropDownIcon />}>
              <Typography>
                <Checkbox checked={reactiveflowlayeractive} onChange={handleReactiveFlowLayerChange} />
                Reactive Flow
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography component="div">
                {reactiveflowlayeractive && (
                  <div style={{ paddingRight: 40 }}>
                    <Slider value={flowfilterreactivevalue} valueLabelDisplay="auto" onChange={handleReactiveFlowRangeFilterChange} getAriaValueText={valuetext} step={10} min={0} max={120} />
                  </div>
                )}
              </Typography>
            </AccordionDetails>
          </Accordion>

          {/* Generation */}
          <Accordion defaultExpanded={false}>
            <AccordionSummary style={accordionSummaryStyle} expandIcon={<ArrowDropDownIcon />}>
              <Typography>
                <Checkbox checked={genlayeractive} onChange={handleGenLayerChange} />
                Generation Power
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography component="div">
                {genlayeractive && (
                  <>
                    <div style={{ paddingRight: 40 }}>
                      <Checkbox checked={genlayercapactive} onChange={handleGenLayerCapChange} />
                      Generation Capacity
                    </div>
                    <div style={{ paddingRight: 40 }}>
                      <Slider value={genfiltervalue} valueLabelDisplay="auto" onChange={handleGenRangeFilterChange} getAriaValueText={valuetext} step={100} min={gendata.minPg} max={gendata.maxPg + 10} />
                    </div>
                    <div style={{ paddingRight: 40 }}>
                      <Multiselect defaultValue={nameSelectItems} data={nameItems} placeholder="Search for generators" onChange={handleGenMultiselect} />
                    </div>
                    <div style={{ width: 300, height: 300, transform: "translate(-1.5vw, 10px)" }}>
                      <Doughnut data={chartdata} options={{ plugins: { legend: { onClick: handleDoughnutClick } } }} />
                    </div>
                  </>
                )}
              </Typography>
            </AccordionDetails>
          </Accordion>

          {/* Load */}
          <Accordion defaultExpanded={false}>
            <AccordionSummary style={accordionSummaryStyle} expandIcon={<ArrowDropDownIcon />}>
              <Typography>
                <Checkbox checked={loadlayeractive} onChange={handleLoadLayerChange} />
                Load loss
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography component="div">
                {loadlayeractive && (
                  <>
                    <div style={{ paddingRight: 40 }}>
                      <Slider value={loadfiltervalue} valueLabelDisplay="auto" onChange={handleLoadRangeFilterChange} getAriaValueText={valuetext} step={100} min={0} max={countyloaddata.maxPd + 10} />
                    </div>
                    <div style={{ paddingRight: 40 }}>
                      <Multiselect defaultValue={countyNameSelectItems} data={countyNameItems} placeholder="Search for counties" onChange={handleCountyMultiselect} />
                    </div>
                  </>
                )}
              </Typography>
            </AccordionDetails>
          </Accordion>

          {/* Voltage */}
          <Accordion defaultExpanded={false}>
            <AccordionSummary style={accordionSummaryStyle} expandIcon={<ArrowDropDownIcon />}>
              <Typography>
                <Checkbox checked={voltagelayeractive} onChange={handleVoltageLayerChange} />
                Voltage
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography component="div">
                {voltagelayeractive && (
                  <>
                    <div style={{ paddingRight: 40 }}>
                      <Slider value={voltagefiltervalue} valueLabelDisplay="auto" onChange={handleVoltageRangeFilterChange} getAriaValueText={valuetext} step={0.01} min={0.89} max={1.11} />
                    </div>
                    <div style={{ paddingRight: 40 }}>
                      <Multiselect defaultValue={countyNameSelectItems} data={countyNameItems} placeholder="Search for counties" onChange={handleCountyMultiselect} />
                    </div>
                  </>
                )}
              </Typography>
            </AccordionDetails>
          </Accordion>

          {/* Areas */}
          <Accordion defaultExpanded={false}>
            <AccordionSummary style={accordionSummaryStyle} expandIcon={<ArrowDropDownIcon />}>
              <Typography>
                <Checkbox checked={arealayeractive} onChange={handleAreaLayerChange} />
                Show Areas
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography component="div">
                {arealayeractive && (
                  <div style={{ paddingRight: 40 }}>
                    <Multiselect defaultValue={areaNameSelectItems} data={areaNameItems} placeholder="Search for areas" onChange={handleAreaMultiselect} />
                  </div>
                )}
              </Typography>
            </AccordionDetails>
          </Accordion>

          {/* Zones */}
          <Accordion defaultExpanded={false}>
            <AccordionSummary style={accordionSummaryStyle} expandIcon={<ArrowDropDownIcon />}>
              <Typography>
                <Checkbox checked={zonelayeractive} onChange={handleZoneLayerChange} />
                Show Zones
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography component="div">
                {zonelayeractive && (
                  <div style={{ paddingRight: 40 }}>
                    <Multiselect defaultValue={zoneNameSelectItems} data={zoneNameItems} placeholder="Search for zones" onChange={handleZoneMultiselect} />
                  </div>
                )}
              </Typography>
            </AccordionDetails>
          </Accordion>
        </div>
      </div>

      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onPatch={patchSettings}
      />

      <ChatBot
        settings={{
          header: { title: "Chat Grid", showAvatar: false },
          general: { embedded: false },
          footer: { text: "", buttons: [] },
          tooltip: { text: "Ask me anything." },
          voice: { disabled: false },
          chatHistory: { storageKey: "example_basic_form" },
          chatButton: { icon: "exago-logo.png" },
        }}
        flow={flow}
      />

      {tooltip && (
        <div className="tooltip" style={tooltip.position}>
          {tooltip.content}
        </div>
      )}
    </div>
  );
}

// ─── Tooltip helper ───────────────────────────────────────────────────────────
function getTooltipState(info) {
  if (!info?.object) return undefined;
  const { x, y, object } = info;
  const position = { left: x, top: y };

  switch (object?.type) {
    case PickingType.LOCATION:
      return { position, content: <div>{object.name}</div> };
    case PickingType.FLOW:
      return {
        position,
        content: (
          <>
            <div>{object.origin.id} → {object.dest.id}</div>
            <div>{object.count}</div>
            <div>{object.loading}</div>
          </>
        ),
      };
    default:
      return undefined;
  }
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const legendWrap = { right: 12, bottom: 12, fontFamily: "Inter, system-ui, sans-serif", fontSize: 13, maxWidth: 220 };
const legendTitle = { fontWeight: 600, marginBottom: 8, color: "#111" };
const row = { display: "flex", alignItems: "center", gap: 8, margin: "4px 0" };
const swatch = { width: 18, height: 12, borderRadius: 3, border: "1px solid rgba(0,0,0,0.2)", flex: "0 0 auto" };
const label = { color: "#222" };
const accordionSummaryStyle = { height: "20px", minHeight: "30px", paddingRight: "40px", paddingLeft: "0px" };

// Generator / network hover tooltip cell styles
const ttHead = { fontWeight: 700, fontSize: 15, marginBottom: 6, borderBottom: "1px solid rgba(255,255,255,0.25)", paddingBottom: 4 };
const ttLabel = { color: "rgba(255,255,255,0.65)", paddingRight: 12, paddingBottom: 3, whiteSpace: "nowrap", fontSize: 12 };
const ttVal = { color: "#fff", fontWeight: 500, paddingBottom: 3, fontSize: 13 };

const btnStyle = (bg) => ({
  padding: "6px 12px",
  background: bg,
  color: "white",
  border: "none",
  borderRadius: 5,
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 12,
  boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
});

// ─── AppContainer ─────────────────────────────────────────────────────────────
function AppContainer() {
  const [selected, setSelected] = useState(DATA_FILES[0]);
  const [casedata, setCasedata] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setErr(null);
      try {
        const res = await fetch(`/data/${selected}`);
        if (!res.ok) throw new Error(`Failed to load ${selected}: ${res.status}`);
        const json = await res.json();
        if (!cancelled) setCasedata(json);
      } catch (e) {
        if (!cancelled) setErr(e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [selected]);

  const derived = useMemo(() => {
    if (!casedata) return null;

    const geodata = casedata["geojsondata"];
    let grid_data = ExtractFirstTimeSlice(geodata);

    const countyloaddata = getCountyNodes(grid_data);
    grid_data = countyloaddata.updatedata;

    const areas = getAreas(casedata);
    const zones = getZones(casedata);

    const [grid_flowdata, grid_flowdata_reactive] = ExtractFlowData(grid_data);

    const Points = getPoints(grid_data);
    const Vcontour = getContours();
    const gendata = getGeneration(grid_data);
    const generation = gendata.Gens;
    const loaddata = getLoad(grid_data);

    const bboxArray = bbox(grid_data);
    const mapcenter = center(grid_data);

    // Compute a view state that fits the full extent of the loaded dataset.
    // Subtract the right-side control panel (300 px) so the data isn't hidden.
    const PANEL_WIDTH = 310;
    const vp = new WebMercatorViewport({
      width: Math.max(window.innerWidth - PANEL_WIDTH, 400),
      height: Math.max(window.innerHeight, 400),
    });
    const [minLng, minLat, maxLng, maxLat] = bboxArray;
    const { longitude: homeLon, latitude: homeLat, zoom: homeZoom } = vp.fitBounds(
      [[minLng, minLat], [maxLng, maxLat]],
      { padding: 60 }
    );
    const homeViewState = {
      longitude: homeLon,
      latitude: homeLat,
      zoom: Math.min(homeZoom, 14),   // cap so we never over-zoom
      pitch: 0,
      bearing: 0,
      maxZoom: 16,
    };

    return {
      geodata, grid_data, areas, zones,
      grid_flowdata, grid_flowdata_reactive,
      Points, Vcontour, gendata, generation,
      loaddata,
      countymaxPd: countyloaddata.maxPd,
      countyload: countyloaddata.data,
      mapcenter, countyloaddata, homeViewState,
    };
  }, [casedata]);

  return (
    <>
      {/* ── Loading overlay ───────────────────────────────────────────────── */}
      {loading && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 99999,
          background: "rgba(15, 23, 42, 0.55)",
          backdropFilter: "blur(4px)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <div style={{
            background: "#fff", borderRadius: 16,
            padding: "36px 48px",
            boxShadow: "0 24px 64px rgba(0,0,0,0.35)",
            display: "flex", flexDirection: "column", alignItems: "center", gap: 20,
            minWidth: 220,
          }}>
            {/* CSS spinner */}
            <div style={{
              width: 52, height: 52,
              border: "5px solid #e2e8f0",
              borderTopColor: "#3b82f6",
              borderRadius: "50%",
              animation: "grid-spin 0.8s linear infinite",
            }} />
            <style>{`@keyframes grid-spin { to { transform: rotate(360deg); } }`}</style>
            <div style={{ fontSize: 16, fontWeight: 600, color: "#1e293b", letterSpacing: 0.2 }}>
              Loading dataset…
            </div>
            <div style={{ fontSize: 13, color: "#64748b" }}>
              Please wait
            </div>
          </div>
        </div>
      )}

      {/* ── Error overlay ────────────────────────────────────────────────── */}
      {err && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 99999,
          background: "rgba(15, 23, 42, 0.55)",
          backdropFilter: "blur(4px)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <div style={{
            background: "#fff", borderRadius: 16,
            padding: "32px 40px",
            boxShadow: "0 24px 64px rgba(0,0,0,0.35)",
            display: "flex", flexDirection: "column", alignItems: "center", gap: 16,
            maxWidth: 360,
          }}>
            <div style={{ fontSize: 36 }}>⚠️</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "#dc2626" }}>Failed to load</div>
            <div style={{ fontSize: 13, color: "#475569", textAlign: "center", wordBreak: "break-word" }}>
              {String(err)}
            </div>
            <button
              onClick={() => window.location.reload()}
              style={{ marginTop: 4, padding: "8px 20px", background: "#3b82f6", color: "#fff", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 600, fontSize: 14 }}
            >
              Reload
            </button>
          </div>
        </div>
      )}

      {derived && (
        <App
          refdata={derived.grid_data}
          refflowdata={derived.grid_flowdata}
          refflowdata_reactive={derived.grid_flowdata_reactive}
          ggdata={derived.geodata}
          gendata={derived.gendata}
          generation={derived.generation}
          areas={derived.areas}
          zones={derived.zones}
          countyload={derived.countyload}
          countyloaddata={derived.countyloaddata}
          countymaxPd={derived.countymaxPd}
          mapcenter={derived.mapcenter}
          homeViewState={derived.homeViewState}
          selected={selected}
          onSelectCase={setSelected}
          onFileUpload={(file) => {
            setLoading(true);
            setErr(null);
            const reader = new FileReader();
            reader.onload = () => {
              try {
                setCasedata(JSON.parse(reader.result));
              } catch (e) {
                setErr(e);
              } finally {
                setLoading(false);
              }
            };
            reader.onerror = () => {
              setErr(new Error("Failed to read file"));
              setLoading(false);
            };
            reader.readAsText(file);
          }}
        />
      )}
    </>
  );
}

export default AppContainer;
