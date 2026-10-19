import { Html, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import { BoxGeometry, type Group, Matrix3, Matrix4, Quaternion, Vector3 } from "three";
import type { Dim3, PackageSpec, Placement, Placements } from "../types";

export type ColorMode = "eta" | "value" | "fragile";

interface Props {
  trucks: Dim3[];
  packages: PackageSpec[];
  placements: Placements;
  colorMode: ColorMode;
  hovered: string | null;
  /** Length of one animation step; a package's move finishes within it. */
  stepSeconds: number;
  onHover: (id: string | null) => void;
}

/** Warm = urgent / valuable, cool = relaxed / cheap. */
function heat(t: number): string {
  return `hsl(${Math.round(220 - 200 * t)}, 70%, 55%)`;
}

export function colorFor(pkg: PackageSpec, mode: ColorMode, all: PackageSpec[]): string {
  if (mode === "fragile") return pkg.fragile ? "hsl(0, 70%, 55%)" : "hsl(210, 15%, 62%)";
  const key = mode === "eta" ? (p: PackageSpec) => -p.eta : (p: PackageSpec) => p.value;
  const values = all.map(key);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  return heat(hi === lo ? 0.5 : (key(pkg) - lo) / (hi - lo));
}

// Domain axes are (x, y, z) with z up; three.js has y up, so y and z swap.
const toScene = (x: number, y: number, z: number) => new Vector3(x, z, y);
const SWAP = [0, 2, 1];

/**
 * Rotation that turns a package from its original orientation into `orientation`
 * (placement.orientation[i] = which original axis ends up along world axis i).
 * A box looks the same mirrored, so an odd permutation is made a proper rotation
 * by flipping one local axis, which lets the transition be a true rotation.
 */
function orientationQuaternion(orientation: Dim3): Quaternion {
  const m = new Matrix3();
  const e = m.elements; // column-major: e[col * 3 + row]
  e.fill(0);
  orientation.forEach((local, world) => {
    e[SWAP[local] * 3 + SWAP[world]] = 1;
  });
  if (m.determinant() < 0) for (let row = 0; row < 3; row++) e[row] = -e[row];
  return new Quaternion().setFromRotationMatrix(new Matrix4().setFromMatrix3(m));
}

const IDENTITY_ORIENTATION: Dim3 = [0, 1, 2];

const TRUCK_GAP = 2;

/** Left edge of each truck when the fleet is laid out side by side along x. */
function truckOffsets(trucks: Dim3[]): number[] {
  const offsets: number[] = [];
  let x = 0;
  for (const [width] of trucks) {
    offsets.push(x);
    x += width + TRUCK_GAP;
  }
  return offsets;
}

/** Where packages wait outside the trucks: shelves along the fleet's +x side. */
function stagingSpots(packages: PackageSpec[], fleetWidth: number, L: number) {
  const W = fleetWidth;
  const gap = 0.6;
  const spots = new Map<string, Vector3>();
  // Shelves are as long as the truck, or longer when many packages would
  // otherwise make the waiting area a thin strip far from the truck.
  const footprint = packages.reduce((sum, p) => sum + (p.dimensions[0] + gap) * (p.dimensions[1] + gap), 0);
  const shelfLength = Math.max(L, Math.sqrt(footprint * 1.3));
  let x = W + 1.5;
  let z = 0;
  let columnWidth = 0;
  let maxZ = 0;
  for (const pkg of packages) {
    const [w, l, h] = pkg.dimensions;
    if (z > 0 && z + l > shelfLength) {
      x += columnWidth + gap;
      z = 0;
      columnWidth = 0;
    }
    spots.set(pkg.id, toScene(x + w / 2, z + l / 2, h / 2));
    z += l + gap;
    maxZ = Math.max(maxZ, z);
    columnWidth = Math.max(columnWidth, w);
  }
  return { spots, maxX: x + columnWidth, maxZ };
}

function WireBox({ size, position, color }: { size: Dim3; position: Dim3; color: string }) {
  const geometry = useMemo(() => new BoxGeometry(...size), [size]);
  return (
    <lineSegments position={position}>
      <edgesGeometry args={[geometry]} />
      <lineBasicMaterial color={color} />
    </lineSegments>
  );
}

/** A timed move: it always runs to completion, so consecutive steps never fight. */
interface Tween {
  elapsed: number;
  delay: number;
  duration: number;
  points: Vector3[];
  lengths: number[];
  total: number;
  from: Quaternion;
  to: Quaternion;
}

interface Motion {
  tween: Tween | null;
  /** Truck the package was last heading to (null = waiting area); undefined before the first render. */
  truck: number | null | undefined;
}

// Within one step the work is staged so each kind of change reads on its own:
// packages leave first, then packages already inside shift or turn, then new ones enter.
const PHASES = {
  exit: { start: 0, share: 0.5 },
  move: { start: 0.25, share: 0.5 },
  enter: { start: 0.5, share: 0.5 },
} as const;

const smooth = (p: number) => p * p * (3 - 2 * p);

function makeTween(
  points: Vector3[],
  from: Quaternion,
  to: Quaternion,
  phase: keyof typeof PHASES,
  stepSeconds: number,
): Tween {
  const lengths = points.slice(1).map((point, i) => point.distanceTo(points[i]));
  const window = stepSeconds * 0.9; // leave a beat before the next step begins
  return {
    elapsed: 0,
    delay: window * PHASES[phase].start,
    duration: window * PHASES[phase].share,
    points,
    lengths,
    total: lengths.reduce((a, b) => a + b, 0),
    from: from.clone(),
    to,
  };
}

function PackageBox({
  pkg,
  placement,
  truckIndex,
  offsetX,
  stage,
  liftY,
  stepSeconds,
  color,
  active,
  dimmed,
  onHover,
}: {
  pkg: PackageSpec;
  placement: Placement | null;
  /** Which truck the placement is in; null when the package is not loaded. */
  truckIndex: number | null;
  /** Where that truck's left edge sits in the scene. */
  offsetX: number;
  stage: Vector3;
  liftY: number;
  stepSeconds: number;
  color: string;
  active: boolean;
  dimmed: boolean;
  onHover: (id: string | null) => void;
}) {
  const group = useRef<Group>(null);
  const motion = useRef<Motion>({ tween: null, truck: undefined });
  // Read at the moment a step starts, so changing the speed never restarts a move.
  const step = useRef(stepSeconds);
  step.current = stepSeconds;
  // The box keeps its real shape; only position and rotation ever change.
  const [d0, d1, d2] = pkg.dimensions;
  const size = useMemo<Dim3>(() => [d0, d2, d1], [d0, d1, d2]);

  const key = placement
    ? `${truckIndex}|${placement.coordinate.join()}|${placement.orientation.join()}`
    : "outside";

  useLayoutEffect(() => {
    const g = group.current;
    if (!g) return;
    const m = motion.current;
    const target = placement
      ? toScene(
          offsetX + placement.coordinate[0] + placement.dimensions[0] / 2,
          placement.coordinate[1] + placement.dimensions[1] / 2,
          placement.coordinate[2] + placement.dimensions[2] / 2,
        )
      : stage;
    const quaternion = orientationQuaternion(placement ? placement.orientation : IDENTITY_ORIENTATION);

    if (m.truck === undefined) {
      // First render: appear in place rather than flying in.
      g.position.copy(target);
      g.quaternion.copy(quaternion);
    } else {
      const here = g.position.clone();
      const crossing = m.truck !== truckIndex;
      const phase = !crossing ? "move" : truckIndex === null ? "exit" : m.truck === null ? "enter" : "move";
      // Leaving, entering, or changing truck: rise clear of the walls, travel across, then settle.
      const points = crossing
        ? [here, new Vector3(here.x, liftY, here.z), new Vector3(target.x, liftY, target.z), target]
        : [here, target];
      m.tween = makeTween(points, g.quaternion, quaternion, phase, step.current);
    }
    m.truck = truckIndex;
  }, [key, stage, liftY]);

  useFrame((_, delta) => {
    const g = group.current;
    const tween = motion.current.tween;
    if (!g || !tween) return;
    tween.elapsed += delta;
    const progress = Math.min(1, Math.max(0, (tween.elapsed - tween.delay) / tween.duration));
    const eased = smooth(progress);

    // Walk the polyline: `eased` of the way along its total length.
    let remaining = eased * tween.total;
    let i = 0;
    while (i < tween.lengths.length - 1 && remaining > tween.lengths[i]) remaining -= tween.lengths[i++];
    const segment = tween.lengths[i] || 1;
    g.position.lerpVectors(tween.points[i], tween.points[i + 1], Math.min(1, remaining / segment));
    g.quaternion.slerpQuaternions(tween.from, tween.to, eased);
    if (progress >= 1) motion.current.tween = null;
  });

  return (
    <group ref={group}>
      <mesh
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(pkg.id);
        }}
        onPointerOut={() => onHover(null)}
      >
        <boxGeometry args={size} />
        <meshStandardMaterial color={color} transparent opacity={dimmed ? 0.25 : active ? 1 : 0.85} />
      </mesh>
      <WireBox size={size} position={[0, 0, 0]} color={active ? "#ffffff" : "#1b1f24"} />
    </group>
  );
}

export default function TruckView({ trucks, packages, placements, colorMode, hovered, stepSeconds, onHover }: Props) {
  const offsets = useMemo(() => truckOffsets(trucks), [trucks]);
  const fleetWidth = offsets[offsets.length - 1] + trucks[trucks.length - 1][0];
  const maxL = Math.max(...trucks.map((t) => t[1]));
  const maxH = Math.max(...trucks.map((t) => t[2]));
  const { spots, maxX, maxZ } = useMemo(
    () => stagingSpots(packages, fleetWidth, maxL),
    [packages, fleetWidth, maxL],
  );
  const liftY = maxH + Math.max(...packages.flatMap((p) => p.dimensions), 1) / 2 + 1;

  const sceneWidth = Math.max(maxX, fleetWidth);
  const sceneDepth = Math.max(maxZ, maxL);
  const reach = Math.max(sceneWidth, sceneDepth, maxH);
  const center: Dim3 = [sceneWidth / 2, maxH / 2, sceneDepth / 2];
  const layout = `${trucks.map((t) => t.join("x")).join("+")}:${sceneWidth}x${sceneDepth}`;
  return (
    <div className="truck-canvas">
      <Canvas
        key={layout}
        // The canvas is much wider than tall, so a wide fleet fits from fairly close in.
        camera={{ position: [center[0] + reach * 0.15, maxH + reach * 0.45, sceneDepth / 2 + reach * 0.85], fov: 40 }}
      >
        <ambientLight intensity={0.9} />
        <directionalLight position={[reach, reach * 2, reach]} intensity={1.2} />
        {trucks.map(([W, L, H], i) => (
          <group key={i}>
            <WireBox size={[W, H, L]} position={[offsets[i] + W / 2, H / 2, L / 2]} color="#8b95a1" />
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[offsets[i] + W / 2, -0.01, L / 2]}>
              <planeGeometry args={[W, L]} />
              <meshBasicMaterial color="#8b95a1" transparent opacity={0.12} />
            </mesh>
            {trucks.length > 1 && (
              <Html position={[offsets[i] + W / 2, H + 0.4, L / 2]} center>
                <span className="truck-label">Truck {i + 1}</span>
              </Html>
            )}
          </group>
        ))}
        {/* Waiting area for packages that are not in any truck. */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[(fleetWidth + 1 + maxX) / 2, -0.01, maxZ / 2]}>
          <planeGeometry args={[maxX - fleetWidth - 1, maxZ]} />
          <meshBasicMaterial color="#8b95a1" transparent opacity={0.06} />
        </mesh>
        {packages.map((pkg) => {
          const truckIndex = placements.findIndex((truck) => truck[pkg.id] != null);
          return (
            <PackageBox
              key={pkg.id}
              pkg={pkg}
              placement={truckIndex >= 0 ? placements[truckIndex][pkg.id] : null}
              truckIndex={truckIndex >= 0 ? truckIndex : null}
              offsetX={truckIndex >= 0 ? offsets[truckIndex] : 0}
              stage={spots.get(pkg.id)!}
              liftY={liftY}
              stepSeconds={stepSeconds}
              color={colorFor(pkg, colorMode, packages)}
              active={hovered === pkg.id}
              dimmed={hovered !== null && hovered !== pkg.id}
              onHover={onHover}
            />
          );
        })}
        <OrbitControls makeDefault target={center} />
      </Canvas>
    </div>
  );
}
