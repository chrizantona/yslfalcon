import {
  ACESFilmicToneMapping,
  BufferGeometry,
  DirectionalLight,
  HemisphereLight,
  Material,
  Mesh,
  OrthographicCamera,
  Scene,
  WebGLRenderer,
} from 'three';
import { createAbstractVehicle } from './abstractVehicle.ts';
import { startBackdropMotion } from './backdropMotion.ts';

export function mountVehicleScene(
  canvas: HTMLCanvasElement,
  onMotionAvailable: (available: boolean) => void,
  onFailure: () => void,
) {
  // Blur is applied only to this canvas in CSS; the UI remains a separate sharp layer.
  const renderer = new WebGLRenderer({
    canvas,
    alpha: true,
    antialias: false,
    powerPreference: 'low-power',
  });
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = ACESFilmicToneMapping;
  const scene = new Scene();
  const vehicle = createAbstractVehicle();
  vehicle.position.y = -0.85;
  scene.add(vehicle);
  scene.add(new HemisphereLight(0xffffff, 0x343434, 2.4));
  const key = new DirectionalLight(0xffffff, 4);
  key.position.set(-3, 7, 5);
  const rim = new DirectionalLight(0xffffff, 3);
  rim.position.set(3, 2, -5);
  scene.add(key, rim);
  const camera = new OrthographicCamera(-4, 4, 3, -3, 0.1, 40);
  camera.position.set(0, 2.6, 9);
  camera.lookAt(0, 0, 0);
  let disposed = false;
  let failed = false;
  let motion: ReturnType<typeof startBackdropMotion> | undefined;

  const fail = () => {
    if (failed || disposed) return;
    failed = true;
    motion?.dispose();
    onMotionAvailable(false);
    onFailure();
  };
  const paint = (angle: number) => {
    if (failed || disposed) return;
    vehicle.rotation.y = angle;
    try {
      renderer.render(scene, camera);
    } catch {
      // A decorative GPU failure must never interrupt the operator's work.
      fail();
    }
  };
  const resize = () => {
    if (failed || disposed) return;
    const { width, height } = canvas.getBoundingClientRect();
    if (!width || !height) return;
    const aspect = width / height;
    const halfWidth = Math.max(3.65, 2.35 * aspect);
    camera.left = -halfWidth;
    camera.right = halfWidth;
    camera.top = halfWidth / aspect;
    camera.bottom = -halfWidth / aspect;
    camera.updateProjectionMatrix();
    // Defocused geometry needs neither Retina resolution nor multi-megapixel buffers.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25, 1600 / width));
    renderer.setSize(width, height, false);
    motion?.redraw();
  };
  const contextLost = (event: Event) => {
    event.preventDefault();
    fail();
  };
  canvas.addEventListener('webglcontextlost', contextLost);
  window.addEventListener('resize', resize, { passive: true });
  resize();
  motion = startBackdropMotion(paint, onMotionAvailable);
  if (failed) motion.dispose();

  return {
    setPaused: (paused: boolean) => motion?.setPaused(paused),
    dispose() {
      if (disposed) return;
      disposed = true;
      motion?.dispose();
      canvas.removeEventListener('webglcontextlost', contextLost);
      window.removeEventListener('resize', resize);
      const geometries = new Set<BufferGeometry>();
      const materials = new Set<Material>();
      vehicle.traverse((object) => {
        if (!(object instanceof Mesh)) return;
        geometries.add(object.geometry);
        for (const material of Array.isArray(object.material)
          ? object.material
          : [object.material]) {
          materials.add(material);
        }
      });
      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((material) => material.dispose());
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
