import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoLayer, PlaceMode, SimState } from "../types";

const STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    esri: {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      attribution: "Esri, Maxar, Earthstar Geographics — published imagery (not simulated)",
    },
  },
  layers: [{ id: "esri", type: "raster", source: "esri" }],
};

function circlePolygon(lat: number, lon: number, radiusM: number, n = 48): GeoJSON.Feature {
  const coords: [number, number][] = [];
  const latR = radiusM / 111320;
  const lonR = radiusM / (111320 * Math.cos((lat * Math.PI) / 180));
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI;
    coords.push([lon + lonR * Math.sin(a), lat + latR * Math.cos(a)]);
  }
  return {
    type: "Feature",
    properties: {},
    geometry: { type: "Polygon", coordinates: [coords] },
  };
}

interface Props {
  state: SimState | null;
  layers: GeoLayer[];
  placeMode: PlaceMode;
  showCoverage: boolean;
  showSensors: boolean;
  showComms: boolean;
  showTrails: boolean;
  onMapClick: (lat: number, lon: number) => void;
  onHover: (lat: number, lon: number) => void;
}

export function MapView({
  state,
  layers,
  placeMode,
  showCoverage,
  showSensors,
  showComms,
  showTrails,
  onMapClick,
  onHover,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const ready = useRef(false);
  const fitted = useRef<string | null>(null);
  const clickRef = useRef(onMapClick);
  const hoverRef = useRef(onHover);
  clickRef.current = onMapClick;
  hoverRef.current = onHover;

  useEffect(() => {
    if (!ref.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: ref.current,
      style: STYLE,
      center: [145.31, -37.52],
      zoom: 11.2,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left");
    map.on("load", () => {
      ready.current = true;
      map.resize();
      ensureLayers(map);
      if (state) applyState(map, state, { showCoverage, showSensors, showComms, showTrails }, layers);
    });
    map.on("click", (e) => clickRef.current(e.lngLat.lat, e.lngLat.lng));
    map.on("mousemove", (e) => hoverRef.current(e.lngLat.lat, e.lngLat.lng));
    mapRef.current = map;
    const ro = new ResizeObserver(() => {
      map.resize();
    });
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      map.remove();
      mapRef.current = null;
      ready.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready.current) return;
    map.getCanvas().style.cursor = placeMode === "pan" ? "" : "crosshair";
  }, [placeMode]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready.current || !state) return;
    applyState(map, state, { showCoverage, showSensors, showComms, showTrails }, layers);
    const key = `${state.bounds.south},${state.bounds.west},${state.bounds.north},${state.bounds.east}`;
    if (fitted.current !== key) {
      fitted.current = key;
      map.fitBounds(
        [
          [state.bounds.west, state.bounds.south],
          [state.bounds.east, state.bounds.north],
        ],
        { padding: 48, maxZoom: 12.5, duration: 400 },
      );
    }
  }, [state, layers, showCoverage, showSensors, showComms, showTrails]);

  return <div ref={ref} className="absolute inset-0 h-full min-h-[240px] w-full" />;
}

function ensureLayers(map: maplibregl.Map) {
  const empty: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };
  const add = (id: string, spec: Record<string, unknown>, before?: string) => {
    if (!map.getSource(id)) {
      map.addSource(id, { type: "geojson", data: empty });
    }
    if (!map.getLayer(id)) {
      map.addLayer({ ...spec, id, source: id } as maplibregl.AddLayerObject, before);
    }
  };
  add("aoi", {
    type: "line",
    paint: { "line-color": "#8b98a5", "line-width": 1.5, "line-dasharray": [2, 2] },
  });
  add("coverage", {
    type: "fill",
    paint: { "fill-color": "#3d8bfd", "fill-opacity": ["get", "o"] },
  });
  add("burned", {
    type: "fill",
    paint: { "fill-color": "#5a2a12", "fill-opacity": 0.45 },
  });
  add("fire-poly", {
    type: "fill",
    paint: {
      "fill-color": [
        "case",
        ["get", "suppressed"],
        "#2f9e6c",
        ["get", "active"],
        "#d4480b",
        "#5a2a12",
      ],
      "fill-opacity": 0.38,
    },
  });
  add("fire-line", {
    type: "line",
    paint: {
      "line-color": [
        "case",
        ["get", "suppressed"],
        "#5ec8b0",
        ["get", "active"],
        "#ff7a3d",
        "#8a5a3a",
      ],
      "line-width": 2,
    },
  });
  add("sensors", {
    type: "fill",
    paint: { "fill-color": "#3d8bfd", "fill-opacity": 0.08 },
  });
  add("suppress", {
    type: "line",
    paint: { "line-color": "#5ec8b0", "line-width": 1.2, "line-dasharray": [2, 2] },
  });
  add("trails", {
    type: "line",
    paint: { "line-color": "#7eb0ff", "line-width": 1.4, "line-opacity": 0.7 },
  });
  add("comms", {
    type: "line",
    paint: { "line-color": "#c9a227", "line-width": 1.2, "line-dasharray": [1, 1] },
  });
  add("ignitions", {
    type: "circle",
    paint: {
      "circle-radius": 5,
      "circle-color": [
        "case",
        ["get", "suppressed"],
        "#2f9e6c",
        "#d4480b",
      ],
      "circle-stroke-color": "#fff",
      "circle-stroke-width": 1,
    },
  });
  add("headings", {
    type: "line",
    paint: { "line-color": "#d6e6ff", "line-width": 2 },
  });
  add("uav-dots", {
    type: "circle",
    paint: {
      "circle-radius": 6,
      "circle-color": "#3d8bfd",
      "circle-stroke-width": 2,
      "circle-stroke-color": ["case", ["==", ["get", "status"], "collided"], "#d4480b", "#e8eef4"],
    },
  });
}

function applyState(
  map: maplibregl.Map,
  state: SimState,
  vis: { showCoverage: boolean; showSensors: boolean; showComms: boolean; showTrails: boolean },
  layers: GeoLayer[],
) {
  const set = (id: string, data: GeoJSON.FeatureCollection | GeoJSON.Feature) => {
    const src = map.getSource(id) as maplibregl.GeoJSONSource | undefined;
    src?.setData(data);
  };

  const aoi = layers.find((l) => l.id === "aoi_envelope" && l.geometry);
  if (aoi?.geometry) {
    set("aoi", { type: "Feature", properties: {}, geometry: aoi.geometry });
  } else {
    const b = state.bounds;
    set("aoi", {
      type: "Feature",
      properties: {},
      geometry: {
        type: "Polygon",
        coordinates: [[
          [b.west, b.south],
          [b.east, b.south],
          [b.east, b.north],
          [b.west, b.north],
          [b.west, b.south],
        ]],
      },
    });
  }

  const firePolys: GeoJSON.Feature[] = [];
  const firePts: GeoJSON.Feature[] = [];
  for (const f of state.fires) {
    firePts.push({
      type: "Feature",
      properties: { detected: f.detected, suppressed: f.suppressed, id: f.fire_id },
      geometry: { type: "Point", coordinates: [f.longitude, f.latitude] },
    });
    if (f.perimeter.length > 2) {
      const ring = f.perimeter.map((p) => [p.longitude, p.latitude] as [number, number]);
      if (ring[0][0] !== ring[ring.length - 1][0]) ring.push(ring[0]);
      firePolys.push({
        type: "Feature",
        properties: { active: f.active, suppressed: f.suppressed },
        geometry: { type: "Polygon", coordinates: [ring] },
      });
    }
  }
  set("fire-poly", { type: "FeatureCollection", features: firePolys });
  set("fire-line", { type: "FeatureCollection", features: firePolys });
  set("ignitions", { type: "FeatureCollection", features: firePts });

  const uavPts: GeoJSON.Feature[] = state.uavs.map((u) => ({
    type: "Feature",
    properties: {
      label: `${u.uav_id} ${Math.round(u.battery_fraction * 100)}%`,
      status: u.status,
    },
    geometry: { type: "Point", coordinates: [u.longitude, u.latitude] },
  }));
  set("uav-dots", { type: "FeatureCollection", features: uavPts });
  const headings: GeoJSON.Feature[] = state.uavs.map((u) => {
    const rad = (u.heading_deg * Math.PI) / 180;
    const m = 350;
    const dlat = (m * Math.cos(rad)) / 111320;
    const dlon = (m * Math.sin(rad)) / (111320 * Math.cos((u.latitude * Math.PI) / 180));
    return {
      type: "Feature",
      properties: {},
      geometry: {
        type: "LineString",
        coordinates: [
          [u.longitude, u.latitude],
          [u.longitude + dlon, u.latitude + dlat],
        ],
      },
    };
  });
  set("headings", { type: "FeatureCollection", features: headings });

  const sensors: GeoJSON.Feature[] = vis.showSensors
    ? state.uavs.map((u) => circlePolygon(u.latitude, u.longitude, u.sensor_range_m))
    : [];
  set("sensors", { type: "FeatureCollection", features: sensors });
  const suppress: GeoJSON.Feature[] = vis.showSensors
    ? state.uavs.map((u) => circlePolygon(u.latitude, u.longitude, u.suppression_range_m ?? 120))
    : [];
  set("suppress", { type: "FeatureCollection", features: suppress });

  const trails: GeoJSON.Feature[] = [];
  if (vis.showTrails && state.trails) {
    for (const pts of Object.values(state.trails)) {
      if (pts.length < 2) continue;
      trails.push({
        type: "Feature",
        properties: {},
        geometry: {
          type: "LineString",
          coordinates: pts.map((p) => [p.longitude, p.latitude]),
        },
      });
    }
  }
  set("trails", { type: "FeatureCollection", features: trails });

  const comms: GeoJSON.Feature[] = [];
  if (vis.showComms) {
    const byId = Object.fromEntries(state.uavs.map((u) => [u.uav_id, u]));
    for (const l of state.communication_links) {
      const a = byId[l.a];
      const b = byId[l.b];
      if (!a || !b) continue;
      comms.push({
        type: "Feature",
        properties: {},
        geometry: {
          type: "LineString",
          coordinates: [
            [a.longitude, a.latitude],
            [b.longitude, b.latitude],
          ],
        },
      });
    }
  }
  set("comms", { type: "FeatureCollection", features: comms });

  const cov: GeoJSON.Feature[] = [];
  if (vis.showCoverage && state.coverage_heatmap.length) {
    const rows = state.coverage_heatmap.length;
    const cols = state.coverage_heatmap[0]?.length ?? 0;
    const dlat = (state.bounds.north - state.bounds.south) / rows;
    const dlon = (state.bounds.east - state.bounds.west) / cols;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = state.coverage_heatmap[r][c];
        if (v <= 0.02) continue;
        const north = state.bounds.north - r * dlat;
        const south = north - dlat;
        const west = state.bounds.west + c * dlon;
        const east = west + dlon;
        cov.push({
          type: "Feature",
          properties: { o: 0.12 + 0.35 * v },
          geometry: {
            type: "Polygon",
            coordinates: [
              [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
              ],
            ],
          },
        });
      }
    }
  }
  set("coverage", { type: "FeatureCollection", features: cov });
}
