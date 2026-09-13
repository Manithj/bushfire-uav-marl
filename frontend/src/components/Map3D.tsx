import { useEffect, useRef } from "react";
import "cesium/Build/Cesium/Widgets/widgets.css";
import type { SimState } from "../types";

interface Props {
  state: SimState | null;
}

export function Map3D({ state }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<{ destroy: () => void; sync: (s: SimState) => void } | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const Cesium = await import("cesium");
      if (cancelled || !ref.current) return;
      const viewer = new Cesium.Viewer(ref.current, {
        terrainProvider: new Cesium.EllipsoidTerrainProvider(),
        baseLayerPicker: false,
        animation: false,
        timeline: false,
        geocoder: false,
        homeButton: false,
        sceneModePicker: false,
        navigationHelpButton: false,
        fullscreenButton: false,
        infoBox: false,
        selectionIndicator: false,
      });
      viewer.imageryLayers.removeAll();
      viewer.imageryLayers.addImageryProvider(
        new Cesium.UrlTemplateImageryProvider({
          url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          credit: "Esri, Maxar — ellipsoid terrain (no DEM bundled)",
        }),
      );
      viewer.scene.globe.enableLighting = false;
      const sync = (s: SimState) => {
        viewer.entities.removeAll();
        for (const f of s.fires) {
          viewer.entities.add({
            position: Cesium.Cartesian3.fromDegrees(f.longitude, f.latitude, 30),
            point: {
              pixelSize: 10,
              color: f.suppressed ? Cesium.Color.LIME : Cesium.Color.ORANGERED,
            },
            label: { text: f.fire_id, font: "11px sans-serif", pixelOffset: new Cesium.Cartesian2(0, -16) },
            ellipse: f.perimeter.length
              ? undefined
              : {
                  semiMajorAxis: f.radius_m,
                  semiMinorAxis: f.radius_m,
                  material: Cesium.Color.ORANGE.withAlpha(0.25),
                },
          });
        }
        for (const u of s.uavs) {
          viewer.entities.add({
            position: Cesium.Cartesian3.fromDegrees(u.longitude, u.latitude, u.altitude_m),
            point: { pixelSize: 8, color: Cesium.Color.DODGERBLUE },
            label: {
              text: `${u.uav_id} ${Math.round(u.battery_fraction * 100)}%`,
              font: "11px sans-serif",
              pixelOffset: new Cesium.Cartesian2(0, -14),
            },
          });
        }
        for (const l of s.communication_links) {
          const a = s.uavs.find((x) => x.uav_id === l.a);
          const b = s.uavs.find((x) => x.uav_id === l.b);
          if (!a || !b) continue;
          viewer.entities.add({
            polyline: {
              positions: Cesium.Cartesian3.fromDegreesArrayHeights([
                a.longitude, a.latitude, a.altitude_m,
                b.longitude, b.latitude, b.altitude_m,
              ]),
              width: 1.5,
              material: Cesium.Color.GOLD.withAlpha(0.7),
            },
          });
        }
      };
      viewerRef.current = { destroy: () => viewer.destroy(), sync };
      if (state) {
        viewer.camera.flyTo({
          destination: Cesium.Cartesian3.fromDegrees(
            (state.bounds.west + state.bounds.east) / 2,
            (state.bounds.south + state.bounds.north) / 2,
            18000,
          ),
          duration: 0.4,
        });
        sync(state);
      }
    })();
    return () => {
      cancelled = true;
      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (state) viewerRef.current?.sync(state);
  }, [state]);

  return (
    <div className="relative h-full w-full">
      <div ref={ref} className="h-full w-full" />
      <div className="absolute bottom-2 left-2 max-w-md bg-ink-900/80 px-2 py-1 text-[10px] text-mute">
        Optional 3D view of the same simulation state. Terrain is an ellipsoid placeholder — no DEM is bundled.
      </div>
    </div>
  );
}
