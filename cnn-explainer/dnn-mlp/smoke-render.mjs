/**
 * Renders the app's bundle in a headless DOM and asserts the network actually
 * drew. A clean Rollup build only proves the code compiles; this proves it runs
 * and produces the SVG the user is meant to see.
 *
 * Uses Node's built-in fetch plus a minimal DOM shim, so no new dependency.
 * Usage: node dnn-mlp/smoke-render.mjs [port]
 */

const port = process.argv[2] ?? '3200';
const base = `http://localhost:${port}`;

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `  ${detail}`}`);
  if (!ok) failures++;
};

// ---------------------------------------------------------------- served files
console.log('Served assets');
const paths = ['/', '/bundle.js', '/bundle.css', '/global.css'];
const bodies = {};

for (const p of paths) {
  try {
    const res = await fetch(base + p);
    bodies[p] = await res.text();
    check(`${p} responds 200`, res.ok, `got ${res.status}`);
  } catch (error) {
    check(`${p} responds 200`, false, error.message);
    bodies[p] = '';
  }
}

// The SPA fallback returns index.html for anything missing, which is how the
// earlier model-loading bug hid itself. Assert each asset is really itself.
check('bundle.js is JavaScript, not the HTML fallback',
  bodies['/bundle.js'].length > 1000 && !bodies['/bundle.js'].trimStart().startsWith('<'));
check('bundle.css is CSS, not the HTML fallback',
  !bodies['/bundle.css'].trimStart().startsWith('<'));
check('index.html references the bundle',
  bodies['/'].includes('bundle.js'));

// ---------------------------------------------------------------- design tokens
console.log('\nStyling');
check('design tokens are defined', bodies['/global.css'].includes('--dnn-accent'));
check('dark theme block is present',
  bodies['/global.css'].includes('prefers-color-scheme: dark'));
check('reduced motion is respected',
  bodies['/global.css'].includes('prefers-reduced-motion'));
check('component styles were extracted',
  bodies['/bundle.css'].length > 500, `${bodies['/bundle.css'].length} bytes`);

// ---------------------------------------------------------------- render
console.log('\nRendering the bundle');

// Minimal DOM shim: enough for Svelte to mount and build an element tree.
/** Svelte sets inline styles via setProperty, so style needs to be an object. */
const makeStyle = () => {
  const props = {};
  return {
    setProperty(k, v) { props[k] = v; },
    removeProperty(k) { delete props[k]; },
    getPropertyValue(k) { return props[k] ?? ''; },
    get cssText() {
      return Object.entries(props).map(([k, v]) => `${k}: ${v}`).join('; ');
    }
  };
};

class Node {
  constructor(name) {
    this.nodeName = name;
    this.childNodes = [];
    this.attributes = {};
    this.style = makeStyle();
    this.parentNode = null;
    this.textContent_ = '';
    this.listeners = {};
  }
  get children() {
    return this.childNodes.filter((c) => c.nodeName !== '#text');
  }
  appendChild(child) {
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }
  insertBefore(child, ref) {
    child.parentNode = this;
    const i = ref ? this.childNodes.indexOf(ref) : -1;
    if (i === -1) this.childNodes.push(child);
    else this.childNodes.splice(i, 0, child);
    return child;
  }
  removeChild(child) {
    const i = this.childNodes.indexOf(child);
    if (i !== -1) this.childNodes.splice(i, 1);
    child.parentNode = null;
    return child;
  }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  getAttribute(k) { return this.attributes[k] ?? null; }
  removeAttribute(k) { delete this.attributes[k]; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn);
  }
  set textContent(v) {
    this.textContent_ = String(v);
    this.childNodes = [];
  }
  get textContent() {
    // Svelte compiles static markup to innerHTML assignments, so that content
    // has to be included here or it looks absent to the assertions below.
    const own = this.innerHTML_
      ? String(this.innerHTML_).replace(/<[^>]*>/g, '')
      : this.textContent_;
    if (this.childNodes.length === 0) return own;
    return own + this.childNodes.map((c) => c.textContent).join('');
  }
  set innerHTML(v) { this.innerHTML_ = String(v); this.childNodes = []; }
  get innerHTML() { return this.innerHTML_ ?? ''; }
  set nodeValue(v) { this.textContent_ = String(v); }
  get nodeValue() { return this.textContent_; }
  set data(v) { this.textContent_ = String(v); }
  get data() { return this.textContent_; }
  get firstChild() { return this.childNodes[0] ?? null; }
  get nextSibling() {
    if (!this.parentNode) return null;
    const i = this.parentNode.childNodes.indexOf(this);
    return this.parentNode.childNodes[i + 1] ?? null;
  }
  contains() { return false; }
  closest() { return null; }
  cloneNode() { return new Node(this.nodeName); }
  getBoundingClientRect() {
    return { width: 900, height: 400, top: 0, left: 0, right: 900, bottom: 400 };
  }
  get clientWidth() { return 900; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  // Svelte looks up <head> this way when injecting component styles.
  getElementsByTagName(name) {
    const target = String(name).toLowerCase();
    const out = [];
    const walk = (node) => {
      if (node.nodeName === target) out.push(node);
      node.childNodes.forEach(walk);
    };
    walk(this);
    return out;
  }
  focus() {}
  blur() {}
  get classList() {
    const self = this;
    return {
      add(...c) { self.attributes.class = [...(self.attributes.class || '').split(' ').filter(Boolean), ...c].join(' '); },
      remove() {},
      toggle() {},
      contains() { return false; }
    };
  }
}

const makeDoc = () => {
  const doc = new Node('#document');
  doc.head = doc.appendChild(new Node('head'));
  doc.body = doc.appendChild(new Node('body'));
  doc.documentElement = doc;
  doc.createElement = (n) => new Node(n.toLowerCase());
  doc.createElementNS = (_ns, n) => new Node(n.toLowerCase());
  doc.createTextNode = (t) => { const n = new Node('#text'); n.textContent_ = String(t); return n; };
  doc.createComment = () => new Node('#comment');
  // Svelte dispatches synthetic events through createEvent in some code paths.
  doc.createEvent = (type) => ({
    type,
    initCustomEvent(name) { this.type = name; },
    initEvent(name) { this.type = name; }
  });
  doc.createDocumentFragment = () => new Node('#fragment');
  doc.querySelector = () => null;
  doc.querySelectorAll = () => [];
  doc.addEventListener = () => {};
  doc.removeEventListener = () => {};
  doc.getElementById = () => null;
  // Resolve document-level tag lookups against the real tree, so <head> is found.
  doc.getElementsByTagName = (name) => {
    const target = String(name).toLowerCase();
    if (target === 'head') return [doc.head];
    if (target === 'body') return [doc.body];
    return Node.prototype.getElementsByTagName.call(doc, target);
  };
  return doc;
};

/*
 * Wrap the document so any DOM method this shim does not implement resolves to
 * a no-op rather than throwing.
 *
 * Implementing Svelte's full DOM surface by hand is an endless game of
 * whack-a-mole; the goal here is only to reach first render and inspect the
 * element tree, so unknown side-effecting calls can safely do nothing.
 */
const autoStub = (target) =>
  new Proxy(target, {
    get(obj, prop) {
      if (prop in obj) return obj[prop];
      // Anything unknown and function-shaped becomes a no-op.
      if (typeof prop === 'string' && /^[a-z]/.test(prop)) {
        return () => undefined;
      }
      return undefined;
    }
  });

const document = autoStub(makeDoc());

// Globals the bundle touches during mount.
globalThis.document = document;
globalThis.window = {
  document,
  addEventListener: () => {},
  removeEventListener: () => {},
  requestAnimationFrame: (fn) => setTimeout(() => fn(Date.now()), 0),
  cancelAnimationFrame: (id) => clearTimeout(id),
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  matchMedia: () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} }),
  navigator: { userAgent: 'node' },
  location: { href: base + '/', pathname: '/' }
};
globalThis.navigator = globalThis.window.navigator;
globalThis.location = globalThis.window.location;
globalThis.requestAnimationFrame = globalThis.window.requestAnimationFrame;
globalThis.cancelAnimationFrame = globalThis.window.cancelAnimationFrame;
globalThis.getComputedStyle = globalThis.window.getComputedStyle;
globalThis.Element = Node;
globalThis.HTMLElement = Node;
globalThis.SVGElement = Node;
globalThis.CustomEvent = class { constructor(t, o = {}) { this.type = t; this.detail = o.detail; } };
globalThis.Event = globalThis.CustomEvent;
// The component uses ResizeObserver; a no-op is enough to reach first render.
globalThis.ResizeObserver = class {
  observe() {} unobserve() {} disconnect() {}
};

let renderError;
try {
  // Executing the IIFE bundle mounts the app into document.body.
  const mod = await import(`data:text/javascript,${encodeURIComponent(bodies['/bundle.js'])}`);
  void mod;
} catch (error) {
  renderError = error;
}

check('bundle executes without throwing', renderError === undefined,
  renderError ? `${renderError.message}` : '');

// ---------------------------------------------------------------- tree assertions
const walk = (node, out = []) => {
  out.push(node);
  node.childNodes.forEach((c) => walk(c, out));
  return out;
};

const all = walk(document.body);
const byName = (name) => all.filter((n) => n.nodeName === name);
const text = document.body.textContent;

console.log('\nRendered output');
check('something was mounted', all.length > 1, `${all.length} nodes`);
check('an SVG canvas exists', byName('svg').length > 0);

// Four layers of 4/5/5/2 neurons = 16 circles, plus header mark and legend.
const circles = byName('circle');
check('neurons rendered as circles', circles.length >= 16, `${circles.length} circles`);

// 4*5 + 5*5 + 5*2 = 55 connections, drawn twice (visible + hit target).
const svgPaths = byName('path');
check('connections rendered as paths', svgPaths.length >= 55, `${svgPaths.length} paths`);

const hitTargets = all.filter((n) => n.getAttribute && n.getAttribute('data-connection'));
check('connection hit targets carry indices', hitTargets.length >= 55,
  `${hitTargets.length} hit targets`);

console.log('\nLabels and content');
for (const label of ['Input', 'Hidden Layer 1', 'Hidden Layer 2', 'Output']) {
  check(`layer label "${label}" is present`, text.includes(label));
}
check('prediction section is present', text.includes('Prediction'));
check('class labels are present', text.includes('Cat') && text.includes('Dog'));
check('legend explains the weight encoding',
  text.includes('negative') && text.includes('positive'));
check('reading guide is present', text.includes('Weights carry influence'));

// ---------------------------------------------------------------- phase 2
console.log('\nPhase 2 controls');
check('run button is present', text.includes('Run forward pass'));
check('reset control is present', text.includes('Reset'));
check('skip control is present', text.includes('Skip to end'));
check('step counter is present', /\d+ steps|Step \d+ of \d+/.test(text));
check('input editor is present', text.includes('Input values'));
check('input presets are present',
  text.includes('All zero') && text.includes('Randomize'));
check('the core formula is shown',
  text.includes('inputs × weights + bias') && text.includes('activation'));

// The weights are random, so the UI must say so rather than implying meaning.
check('untrained state is disclosed', text.includes('Untrained'));

const rangeInputs = all.filter((n) =>
  n.nodeName === 'input' && n.getAttribute('type') === 'range');
check('one slider per input', rangeInputs.length === 4, `${rangeInputs.length} sliders`);

const numberInputs = all.filter((n) =>
  n.nodeName === 'input' && n.getAttribute('type') === 'number');
check('one number field per input', numberInputs.length === 4,
  `${numberInputs.length} fields`);

check('sliders are labelled',
  rangeInputs.every((n) => (n.getAttribute('aria-label') || '').startsWith('Input')));

// Playback status must be announced for screen readers.
const liveRegions = all.filter((n) => n.getAttribute && n.getAttribute('aria-live'));
check('playback status is announced', liveRegions.length > 0);

// The diagram must show live numbers, not placeholders.
const hasPercent = /\d+%/.test(text);
check('computed percentages are displayed', hasPercent);
check('no unresolved template placeholders',
  !text.includes('undefined') && !text.includes('NaN'));

console.log(failures === 0 ? '\nAll checks passed.' : `\n${failures} check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
