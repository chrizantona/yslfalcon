import {
  BufferGeometry,
  CatmullRomCurve3,
  CylinderGeometry,
  Float32BufferAttribute,
  Group,
  LatheGeometry,
  Mesh,
  MeshStandardMaterial,
  TorusGeometry,
  TubeGeometry,
  Vector2,
  Vector3,
} from 'three';

// X points toward the nose; the tires rest on Y = 0.
type Section = readonly [x: number, bottom: number, top: number, halfWidth: number];

function interpolate(a: number, b: number, c: number, d: number, t: number): number {
  return b + 0.5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
}

function shell(
  sections: readonly Section[],
  wheelArches = false,
  rings = 96,
  sides = 48,
): BufferGeometry {
  const positions: number[] = [];
  const indices: number[] = [];

  for (let ring = 0; ring <= rings; ring += 1) {
    const position = (ring / rings) * (sections.length - 1);
    const index = Math.min(Math.floor(position), sections.length - 2);
    const fraction = position - index;
    const a = sections[Math.max(0, index - 1)];
    const b = sections[index];
    const c = sections[index + 1];
    const d = sections[Math.min(sections.length - 1, index + 2)];
    const [x, bottom, top, width] = b.map((_, component) =>
      interpolate(a[component], b[component], c[component], d[component], fraction),
    );

    for (let side = 0; side < sides; side += 1) {
      const angle = (side / sides) * Math.PI * 2;
      const sine = Math.sin(angle);
      const cosine = Math.cos(angle);
      const z = Math.sign(cosine) * Math.pow(Math.abs(cosine), 0.66) * width;
      let y =
        (top + bottom) / 2 +
        (Math.sign(sine) * Math.pow(Math.abs(sine), 0.78) * (top - bottom)) / 2;

      if (wheelArches) {
        const distance = Math.min(Math.abs(x - 1.8), Math.abs(x + 1.8));
        if (distance < 0.57) {
          const arch = 0.48 + Math.sqrt(0.57 ** 2 - distance ** 2);
          const blend = Math.min(1, Math.max(0, (Math.abs(z) / width - 0.62) / 0.3));
          y +=
            (Math.max(0, arch - bottom) * blend * blend * (3 - 2 * blend) * (top - y)) /
            (top - bottom);
        }
      }
      positions.push(x, y, z);
    }
  }

  for (let ring = 0; ring < rings; ring += 1) {
    for (let side = 0; side < sides; side += 1) {
      const a = ring * sides + side;
      const b = ring * sides + ((side + 1) % sides);
      indices.push(a, a + sides, b, a + sides, b + sides, b);
    }
  }

  // Close both ends so the silhouette remains a volume during a full rotation.
  for (const end of [0, rings]) {
    const section = sections[end === 0 ? 0 : sections.length - 1];
    const center = positions.length / 3;
    positions.push(section[0], (section[1] + section[2]) / 2, 0);
    for (let side = 0; side < sides; side += 1) {
      const a = end * sides + side;
      const b = end * sides + ((side + 1) % sides);
      indices.push(center, end === 0 ? a : b, end === 0 ? b : a);
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

export function createAbstractVehicle(): Group {
  const vehicle = new Group();
  vehicle.name = 'Abstract concept coupe';
  const body = new MeshStandardMaterial({ color: 0xb7b7b7, metalness: 0.58, roughness: 0.32 });
  const glass = new MeshStandardMaterial({ color: 0x202020, metalness: 0.35, roughness: 0.2 });
  const rubber = new MeshStandardMaterial({ color: 0x151515, roughness: 0.94 });
  const darkMetal = new MeshStandardMaterial({ color: 0x303030, metalness: 0.7, roughness: 0.42 });
  const alloy = new MeshStandardMaterial({ color: 0x9c9c9c, metalness: 0.82, roughness: 0.26 });
  const lights = new MeshStandardMaterial({
    color: 0xeeeeee,
    emissive: 0xdddddd,
    emissiveIntensity: 0.6,
    roughness: 0.28,
  });

  const add = (geometry: BufferGeometry, material: MeshStandardMaterial, parent = vehicle) => {
    const mesh = new Mesh(geometry, material);
    parent.add(mesh);
    return mesh;
  };

  add(
    shell(
      [
        [-2.88, 0.61, 0.89, 0.73],
        [-2.64, 0.42, 1.08, 0.94],
        [-1.82, 0.34, 1.23, 1.04],
        [-0.84, 0.34, 1.16, 0.98],
        [0.35, 0.35, 1.1, 0.98],
        [1.8, 0.37, 1.11, 1.03],
        [2.57, 0.48, 0.98, 0.91],
        [2.88, 0.65, 0.85, 0.73],
      ],
      true,
    ),
    body,
  );

  add(
    shell([
      [-1.97, 1.03, 1.17, 0.72],
      [-1.45, 1.04, 1.53, 0.8],
      [-0.8, 1.06, 1.79, 0.79],
      [-0.15, 1.06, 1.81, 0.75],
      [0.48, 1.05, 1.55, 0.73],
      [1.17, 1.02, 1.12, 0.77],
    ]),
    glass,
  );

  add(
    shell([
      [-1.21, 0.35, 0.43, 0.9],
      [-0.65, 0.34, 0.43, 1],
      [0.67, 0.35, 0.44, 1],
      [1.22, 0.38, 0.45, 0.92],
    ]),
    darkMetal,
  );

  const tire = new LatheGeometry(
    [
      new Vector2(0.3, -0.14),
      new Vector2(0.39, -0.14),
      new Vector2(0.45, -0.105),
      new Vector2(0.48, -0.055),
      new Vector2(0.48, 0.055),
      new Vector2(0.45, 0.105),
      new Vector2(0.39, 0.14),
      new Vector2(0.3, 0.14),
      new Vector2(0.3, -0.14),
    ],
    48,
  ).rotateX(Math.PI / 2);
  const wheelFace = new CylinderGeometry(0.326, 0.326, 0.255, 40).rotateX(Math.PI / 2);
  const rim = new TorusGeometry(0.324, 0.022, 8, 48);
  const hub = new CylinderGeometry(0.085, 0.085, 0.028, 24).rotateX(Math.PI / 2);
  const spoke = shell(
    [
      [0.04, -0.038, 0.038, 0.012],
      [0.2, -0.032, 0.032, 0.013],
      [0.3, -0.019, 0.019, 0.01],
    ],
    false,
    8,
    8,
  );

  for (const x of [-1.8, 1.8]) {
    for (const side of [-1, 1]) {
      const wheel = new Group();
      wheel.position.set(x, 0.48, side * 0.96);
      vehicle.add(wheel);
      add(tire, rubber, wheel);
      add(wheelFace, darkMetal, wheel);
      add(rim, alloy, wheel).position.z = side * 0.143;
      add(hub, alloy, wheel).position.z = side * 0.151;
      for (let index = 0; index < 5; index += 1) {
        const mesh = add(spoke, alloy, wheel);
        mesh.rotation.z = (index / 5) * Math.PI * 2 + 0.15;
        mesh.position.z = side * 0.145;
      }
    }
  }

  const accent = (points: readonly (readonly [number, number, number])[]) =>
    add(
      new TubeGeometry(
        new CatmullRomCurve3(points.map(([x, y, z]) => new Vector3(x, y, z))),
        24,
        0.017,
        8,
        false,
      ),
      lights,
    );

  for (const side of [-1, 1]) {
    accent([
      [2.865, 0.823, side * 0.18],
      [2.852, 0.835, side * 0.51],
      [2.746, 0.883, side * 0.77],
    ]);
    accent([
      [-2.868, 0.868, side * 0.13],
      [-2.852, 0.883, side * 0.47],
      [-2.76, 0.936, side * 0.76],
    ]);
  }

  return vehicle;
}
