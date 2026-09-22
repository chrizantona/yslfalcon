import { useEffect, useRef, useState } from 'react';
import type { mountVehicleScene } from '../visuals/vehicleScene';

type VehicleScene = ReturnType<typeof mountVehicleScene>;

export function VehicleBackdrop({
  staticVariant,
  paused,
  onMotionAvailable,
}: {
  staticVariant: boolean;
  paused: boolean;
  onMotionAvailable: (available: boolean) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const scene = useRef<VehicleScene | null>(null);
  const pausedRef = useRef(paused);
  const [fallback, setFallback] = useState(staticVariant);

  useEffect(() => {
    pausedRef.current = paused;
    scene.current?.setPaused(paused);
  }, [paused]);

  useEffect(() => {
    const element = canvas.current;
    if (staticVariant || !element) return;
    let cancelled = false;
    let failed = false;
    let mounted: VehicleScene | null = null;
    const fail = () => {
      if (cancelled) return;
      failed = true;
      setFallback(true);
      onMotionAvailable(false);
      mounted?.dispose();
    };
    // Keep the service entry bundle independent from the optional 3D renderer.
    void import('../visuals/vehicleScene')
      .then(({ mountVehicleScene }) => {
        if (cancelled) return;
        mounted = mountVehicleScene(element, onMotionAvailable, fail);
        scene.current = mounted;
        if (failed) mounted.dispose();
        else mounted.setPaused(pausedRef.current);
      })
      .catch(fail);
    return () => {
      cancelled = true;
      mounted?.dispose();
      scene.current = null;
    };
  }, [staticVariant, onMotionAvailable]);

  return (
    <div
      className={`vehicle-backdrop${fallback ? ' vehicle-backdrop-static' : ''}`}
      aria-hidden="true"
    >
      {!staticVariant && (
        <canvas ref={canvas} hidden={fallback} className="vehicle-backdrop-canvas" />
      )}
    </div>
  );
}
