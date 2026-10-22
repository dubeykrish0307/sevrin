// The orb: arc-reactor style rings drawn with three.js. It turns slowly when
// idle, faster while listening and thinking, and while SEVRIN speaks the
// outer arcs swell with the voice.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const C_HOT = 0xdff4ff;
const C_BLUE = 0x2e9eff;
const C_CYAN = 0x6ee7ff;

// per state: brightness, spin speed, halo and core strength
const PALETTE = {
  idle:      { bright: 0.4, spin: 0.06, halo: 0.2,  core: 0.45 },
  listening: { bright: 0.8, spin: 0.3,  halo: 0.35, core: 0.85 },
  thinking:  { bright: 0.9, spin: 0.7,  halo: 0.4,  core: 0.9 },
  speaking:  { bright: 1.0, spin: 0.45, halo: 0.45, core: 1.0 },
};

// each reply settles the core on one of these; kept to a narrow blue/violet range
const ANSWER_HUES = [0x2e9eff, 0x3ad0ff, 0x6ee7ff, 0x4f8bff, 0x8a9bff, 0x36c8d8];
const OUTER_ARCS = 12;

function additive(color, opacity) {
  return new THREE.MeshBasicMaterial({
    color, transparent: true, opacity,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
  });
}

function arc(radius, tube, color, opacity, start, length) {
  const mesh = new THREE.Mesh(new THREE.TorusGeometry(radius, tube, 12, 128, length), additive(color, opacity));
  mesh.rotation.z = start;
  mesh.userData = { baseRadius: radius, push: 0, target: 0 };
  return mesh;
}

function tickRing(count, radius, width, height, color, opacity) {
  const ticks = new THREE.InstancedMesh(new THREE.BoxGeometry(width, height, 0.001), additive(color, opacity), count);
  const dummy = new THREE.Object3D();
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2;
    dummy.position.set(Math.cos(a) * radius, Math.sin(a) * radius, 0);
    dummy.rotation.z = a;
    dummy.updateMatrix();
    ticks.setMatrixAt(i, dummy.matrix);
  }
  return ticks;
}

// radial gradient that fades to fully transparent, so glows have no hard edge
function glowTexture() {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.25, 'rgba(200,240,255,0.9)');
  g.addColorStop(0.55, 'rgba(46,158,255,0.35)');
  g.addColorStop(1, 'rgba(46,158,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

function glowSprite(texture, color, opacity, scale) {
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, color, transparent: true, opacity,
    blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  sprite.scale.set(scale, scale, 1);
  return sprite;
}

export function createReactor(container) {
  let width = container.clientWidth || 400;
  let height = container.clientHeight || 400;

  const scene = new THREE.Scene();
  scene.background = null;

  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 0, 9.0);   // far enough back that fully swollen outer arcs stay in frame
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, premultipliedAlpha: false });
  renderer.setClearColor(0x000000, 0);
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.5));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  container.appendChild(renderer.domElement);

  // A transparent scene plus OutputPass keeps the bloom from drawing its
  // whole render target as a visible box. Low strength, high threshold:
  // only the brightest edges glow.
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(width, height), 0.28, 0.5, 0.6);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  const reactor = new THREE.Group();
  scene.add(reactor);

  const rings = [
    [0.52, 0.016, C_HOT, 0.9],
    [0.78, 0.008, C_CYAN, 0.6],
    [1.05, 0.014, C_BLUE, 0.75],
    [1.5, 0.008, C_CYAN, 0.55],
  ].map(([r, tube, color, opacity]) => new THREE.Mesh(new THREE.TorusGeometry(r, tube, 12, 200), additive(color, opacity)));
  reactor.add(...rings);

  // six thick coil segments that only rotate
  const coils = new THREE.Group();
  for (let i = 0; i < 6; i++) {
    coils.add(arc(1.65, 0.05, C_BLUE, 0.85, (i / 6) * Math.PI * 2 + 0.12, (Math.PI * 2 / 6) - 0.24));
  }
  // finer outer arcs that swell outwards while speaking
  const outer = new THREE.Group();
  for (let i = 0; i < OUTER_ARCS; i++) {
    outer.add(arc(2.2, 0.012, C_CYAN, 0.7, (i / OUTER_ARCS) * Math.PI * 2 + 0.04, (Math.PI * 2 / OUTER_ARCS) - 0.09));
  }
  reactor.add(coils, outer);

  const ticks = tickRing(60, 1.28, 0.09, 0.02, C_CYAN, 0.55);
  const fineTicks = tickRing(120, 1.9, 0.05, 0.012, C_BLUE, 0.4);
  reactor.add(ticks, fineTicks);

  const glow = glowTexture();
  const core = new THREE.Mesh(
    new THREE.CircleGeometry(0.4, 96),
    new THREE.MeshBasicMaterial({
      color: C_HOT, transparent: true, opacity: 1.0, blending: THREE.AdditiveBlending, depthWrite: false,
    }),
  );
  const coreGlow = glowSprite(glow, 0xffffff, 0.7, 1.6);
  // soft halo behind it all; scaled up much more it shows as a lit square
  const halo = glowSprite(glow, 0x1c6bb0, 0.18, 4.2);
  halo.position.z = -0.5;
  reactor.add(core, coreGlow, halo);

  let state = 'idle';
  let amplitude = 0;
  let smoothed = 0;
  let flashLevel = 0;
  let brightness = 0.5;
  let t = 0;
  let framesToNextFlare = 0;
  let hueIndex = 2;
  const answerColor = new THREE.Color(0x6ee7ff);
  const tmpColor = new THREE.Color();

  function animate() {
    requestAnimationFrame(animate);
    t += 0.016;
    smoothed += (amplitude - smoothed) * 0.25;
    flashLevel = Math.max(0, flashLevel - 0.02);

    const p = PALETTE[state] || PALETTE.idle;
    brightness += (p.bright + flashLevel * 0.7 - brightness) * 0.08;

    // The middle layers each turn at their own speed and direction. The
    // innermost ring, the core and the outer arcs stay put; the outer arcs
    // need a still frame for the flares to read against.
    coils.rotation.z += 0.012 * p.spin;
    ticks.rotation.z += 0.006 * p.spin;
    fineTicks.rotation.z -= 0.009 * p.spin;
    rings[1].rotation.z += 0.008 * p.spin;
    rings[2].rotation.z -= 0.005 * p.spin;
    rings[3].rotation.z += 0.003 * p.spin;

    const energy = brightness + smoothed * 0.8;
    rings.forEach((r) => {
      r.material.opacity = Math.min(1, r.material.opacity + (energy * 0.7 - r.material.opacity) * 0.1);
    });

    // While speaking, every 8-16 frames one random outer arc eases outwards
    // by an amount set by the voice level, often with a smaller swell on a
    // neighbour. They ease back on their own.
    const speaking = state === 'speaking' && smoothed > 0.02;
    framesToNextFlare -= 1;
    if (speaking && framesToNextFlare <= 0) {
      const arcs = outer.children;
      const idx = Math.floor(Math.random() * arcs.length);
      arcs[idx].userData.target = 0.28 + smoothed * 0.85;
      if (Math.random() < 0.55) {
        const dir = Math.random() < 0.5 ? 1 : -1;
        const neighbour = arcs[(idx + dir + arcs.length) % arcs.length];
        neighbour.userData.target = Math.max(neighbour.userData.target, 0.14 + smoothed * 0.45);
      }
      framesToNextFlare = 8 + Math.floor(Math.random() * 9);
    }

    outer.children.forEach((a, i) => {
      const d = a.userData;
      d.target *= 0.955;
      // slow breathing per arc, so the ring is never perfectly still
      const breathe = (Math.sin(t * 0.7 + i * 1.7) * 0.5 + 0.5) * 0.03;
      d.push += (d.target + breathe - d.push) * 0.07;
      a.scale.setScalar(1 + d.push / d.baseRadius);
      a.material.opacity = Math.min(1, 0.5 + energy * 0.25 + d.push * 0.5);
    });

    coils.children.forEach((a) => {
      a.material.opacity = Math.min(1, 0.55 + energy * 0.3 + smoothed * 0.2);
    });

    const corePulse = state === 'speaking'
      ? p.core + smoothed * 0.8
      : p.core + Math.sin(t * (state === 'idle' ? 0.8 : 2.5)) * 0.12;
    core.material.opacity = Math.min(1, corePulse);
    core.scale.setScalar(1 + smoothed * 0.4 + flashLevel * 0.3);

    // the core glides towards this reply's hue and drifts a little with the voice
    tmpColor.copy(answerColor).offsetHSL(0.06 * Math.sin(t * 1.7) * smoothed, 0, 0);
    core.material.color.lerp(tmpColor, 0.04);
    coreGlow.material.color.lerp(tmpColor, 0.04);
    coreGlow.material.opacity = Math.min(1, (p.core + smoothed * 0.6) * 0.7);
    coreGlow.scale.setScalar(1.6 * (1 + smoothed * 0.25));

    halo.material.opacity = Math.min(0.5, p.halo * 0.6 + smoothed * 0.15 + flashLevel * 0.15);

    // slight tilt for some depth
    reactor.rotation.x = Math.sin(t * 0.2) * 0.06;
    reactor.rotation.y = Math.sin(t * 0.15) * 0.08;

    composer.render();
  }

  window.addEventListener('resize', () => {
    width = container.clientWidth || 400;
    height = container.clientHeight || 400;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    composer.setSize(width, height);
    bloom.setSize(width, height);
  });

  animate();

  return {
    setState(value) { state = value; },
    setAmplitude(value) { amplitude = value; },
    flash() { flashLevel = 1.0; },
    // move the core to a different hue for the next reply
    nextHue() {
      let next = hueIndex;
      while (next === hueIndex) next = Math.floor(Math.random() * ANSWER_HUES.length);
      hueIndex = next;
      answerColor.setHex(ANSWER_HUES[next]);
    },
  };
}
