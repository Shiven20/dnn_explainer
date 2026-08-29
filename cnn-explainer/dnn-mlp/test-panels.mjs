/**
 * Verifies the Phase 3 panels actually render when something is selected.
 *
 * The plain smoke test only ever sees the default state, where no neuron or
 * connection is selected -- so it never exercises the inspection panels at all.
 * This drives the real stores, then renders the real components against them.
 *
 * Usage: npm run test:panels
 */

// ---------------------------------------------------------------- DOM shim
/*
 * Same approach as smoke-render.mjs: a minimal tree plus a Proxy that turns any
 * unimplemented DOM method into a no-op, because Svelte's DOM surface is far
 * larger than these tests need.
 */
const makeStyle = () => {
  const props = {};
  return {
    setProperty(k, v) { props[k] = v; },
    removeProperty(k) { delete props[k]; },
    getPropertyValue(k) { return props[k] ?? ''; }
  };
};

class El {
  constructor(name) {
    this.nodeName = name;
    this.childNodes = [];
    this.attributes = {};
    this.style = makeStyle();
    this.parentNode = null;
    this.textContent_ = '';
  }
  appendChild(c) { c.parentNode = this; this.childNodes.push(c); return c; }
  insertBefore(c, ref) {
    c.parentNode = this;
    const i = ref ? this.childNodes.indexOf(ref) : -1;
    if (i === -1) this.childNodes.push(c); else this.childNodes.splice(i, 0, c);
    return c;
  }
  removeChild(c) {
    const i = this.childNodes.indexOf(c);
    if (i !== -1) this.childNodes.splice(i, 1);
    c.parentNode = null;
    return c;
  }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  getAttribute(k) { return this.attributes[k] ?? null; }
  removeAttribute(k) { delete this.attributes[k]; }
  addEventListener() {}
  removeEventListener() {}
  set textContent(v) { this.textContent_ = String(v); this.childNodes = []; }
  get textContent() {
    const own = this.innerHTML_
      ? String(this.innerHTML_).replace(/<[^>]*>/g, ' ')
      : this.textContent_;
    if (this.childNodes.length === 0) return own;
    return own + this.childNodes.map((c) => c.textContent).join('');
  }
  set innerHTML(v) { this.innerHTML_ = String(v); this.childNodes = []; }
  get innerHTML() { return this.innerHTML_ ?? ''; }
  set nodeValue(v) { this.textContent_ = String(v); }
  set data(v) { this.textContent_ = String(v); }
  get firstChild() { return this.childNodes[0] ?? null; }
  get nextSibling() {
    if (!this.parentNode) return null;
    const i = this.parentNode.childNodes.indexOf(this);
    return this.parentNode.childNodes[i + 1] ?? null;
  }
  get classList() {
    const self = this;
    return {
      add(...c) {
        self.attributes.class =
          [...(self.attributes.class || '').split(' ').filter(Boolean), ...c].join(' ');
      },
      remove() {}, toggle() {}, contains() { return false; }
    };
  }
  getElementsByTagName(name) {
    const target = String(name).toLowerCase();
    const out = [];
    const walk = (n) => { if (n.nodeName === target) out.push(n); n.childNodes.forEach(walk); };
    walk(this);
    return out;
  }
  getBoundingClientRect() { return { width: 900, height: 400, top: 0, left: 0 }; }
  get clientWidth() { return 900; }
  contains() { return false; }
  closest() { return null; }
  cloneNode() { return new El(this.nodeName); }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  focus() {} blur() {}
}

const autoStub = (target) =>
  new Proxy(target, {
    get(obj, prop) {
      if (prop in obj) return obj[prop];
      if (typeof prop === 'string' && /^[a-z]/.test(prop)) return () => undefined;
      return undefined;
    }
  });

const doc = new El('#document');
doc.head = doc.appendChild(new El('head'));
doc.body = doc.appendChild(new El('body'));
doc.createElement = (n) => new El(String(n).toLowerCase());
doc.createElementNS = (_ns, n) => new El(String(n).toLowerCase());
doc.createTextNode = (t) => { const n = new El('#text'); n.textContent_ = String(t); return n; };
doc.createComment = () => new El('#comment');
doc.createEvent = (t) => ({ type: t, initCustomEvent() {}, initEvent() {} });
doc.getElementsByTagName = (name) => {
  const target = String(name).toLowerCase();
  if (target === 'head') return [doc.head];
  if (target === 'body') return [doc.body];
  return El.prototype.getElementsByTagName.call(doc, target);
};

const document = autoStub(doc);
globalThis.document = document;
globalThis.window = autoStub({
  document,
  addEventListener: () => {},
  removeEventListener: () => {},
  requestAnimationFrame: (fn) => setTimeout(() => fn(Date.now()), 0),
  cancelAnimationFrame: (id) => clearTimeout(id),
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} })
});
globalThis.requestAnimationFrame = globalThis.window.requestAnimationFrame;
globalThis.cancelAnimationFrame = globalThis.window.cancelAnimationFrame;
globalThis.Element = El;
globalThis.HTMLElement = El;
globalThis.SVGElement = El;
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };

// ---------------------------------------------------------------- harness
let failures = 0;
let checks = 0;
const check = (label, ok, detail = '') => {
  checks++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `  ${detail}`}`);
  if (!ok) failures++;
};

/*
 * The components are .svelte files, which Node cannot import directly. Compile
 * them on the fly with the same Svelte version the app builds with, so this
 * tests the real components rather than a reimplementation.
 */
const { compile } = await import('svelte/compiler');
const fs = await import('fs');
const path = await import('path');

const SRC = path.join(process.cwd(), 'src');
const cache = new Map();
const written = [];

/*
 * Compile a .svelte file to a sibling .compiled.mjs and return its path.
 *
 * Writing next to the original (rather than into a temp dir or a data URL) means
 * relative imports like '../stores.js' and 'svelte/internal' resolve exactly as
 * they do in the real build, with no path rewriting beyond .svelte references.
 * The generated files are removed in the cleanup step below.
 */
const compileComponent = (absPath) => {
  if (cache.has(absPath)) return cache.get(absPath);

  const { js } = compile(fs.readFileSync(absPath, 'utf8'), {
    filename: path.basename(absPath),
    generate: 'dom',
    css: false
  });

  let code = js.code;

  // Child components need to point at their compiled siblings.
  const childRefs = [...code.matchAll(/from\s+['"](\.[^'"]+\.svelte)['"]/g)]
    .map((m) => m[1]);

  for (const rel of [...new Set(childRefs)]) {
    compileComponent(path.resolve(path.dirname(absPath), rel));
    code = code.split(rel).join(`${rel}.compiled.mjs`);
  }

  const outPath = `${absPath}.compiled.mjs`;
  fs.writeFileSync(outPath, code);
  written.push(outPath);
  cache.set(absPath, outPath);
  return outPath;
};

const mount = async (relPath, props) => {
  const compiled = compileComponent(path.join(SRC, relPath));
  const mod = await import(`file://${compiled.split(path.sep).join('/')}`);
  const target = new El('div');
  const instance = new mod.default({ target, props });
  return { instance, target };
};

/** Remove the generated files, so nothing compiled leaks into the source tree. */
const cleanup = () => {
  for (const file of written) {
    try { fs.unlinkSync(file); } catch { /* already gone */ }
  }
};

process.on('exit', cleanup);

const textOf = (node) => node.textContent.replace(/\s+/g, ' ').trim();

// ---------------------------------------------------------------- stores
const stores = await import('../src/dnn/stores.js');
const { forwardPass, explainNeuron } = await import('../src/dnn/engine/forwardPass.js');

let net;
stores.activatedNetwork.subscribe((v) => { net = v; });

// ---------------------------------------------------------------- neuron panel
console.log('Neuron details panel');

// A hidden neuron: should show the full weighted sum breakdown.
const hidden = await mount('dnn/components/NeuronDetails.svelte', {
  layerIndex: 1,
  index: 0
});
const hiddenText = textOf(hidden.target);

check('renders the layer name', hiddenText.includes('Hidden Layer 1'), hiddenText.slice(0, 90));
check('names the neuron', hiddenText.includes('Neuron 1'));
check('shows the weighted sum section', hiddenText.includes('Weighted sum'));
check('shows the bias row', hiddenText.includes('bias'));
check('shows the activation section', hiddenText.includes('Activation'));
check('shows the sum of products', hiddenText.includes('sum of products'));

// The numbers on screen must be the network's own.
const expected = explainNeuron(net, 1, 0);
check('displays the real z value',
  hiddenText.includes(expected.preActivation.toFixed(2).replace('-0.00', '0.00')) ||
  hiddenText.includes(expected.preActivation.toFixed(3)),
  `looking for z=${expected.preActivation.toFixed(3)}`);

// One row per incoming connection, each with a clickable weight chip.
const hiddenButtons = [];
(function walk(n) { if (n.nodeName === 'button') hiddenButtons.push(n); n.childNodes.forEach(walk); })(hidden.target);
// 4 source buttons + 4 weight chips + close = 9 minimum.
check('renders a control per term plus a close button',
  hiddenButtons.length >= 9, `${hiddenButtons.length} buttons`);

const weightChips = hiddenButtons.filter((b) =>
  (b.getAttribute('aria-label') || '').startsWith('Weight'));
check('weight chips are focusable buttons', weightChips.length === 4,
  `${weightChips.length} chips`);
check('weight chips describe their action',
  weightChips.every((c) => (c.getAttribute('aria-label') || '').includes('click to edit')));

hidden.instance.$destroy();

// An input neuron: no weighted sum, but a direct value editor.
const inputPanel = await mount('dnn/components/NeuronDetails.svelte', {
  layerIndex: 0,
  index: 1
});
const inputText = textOf(inputPanel.target);
check('input neuron panel explains it holds a value',
  inputText.includes('does not compute'), inputText.slice(0, 90));
check('input neuron panel omits the weighted sum',
  !inputText.includes('Weighted sum'));
const inputRanges = [];
(function walk(n) {
  if (n.nodeName === 'input' && n.getAttribute('type') === 'range') inputRanges.push(n);
  n.childNodes.forEach(walk);
})(inputPanel.target);
check('input neuron panel offers a slider', inputRanges.length === 1);
inputPanel.instance.$destroy();

// An output neuron: softmax explanation instead of a curve.
const outputPanel = await mount('dnn/components/NeuronDetails.svelte', {
  layerIndex: 3,
  index: 0
});
const outputText = textOf(outputPanel.target);
check('output neuron panel mentions softmax', outputText.includes('Softmax'));
check('output neuron panel shows its share', /%/.test(outputText));
outputPanel.instance.$destroy();

// ---------------------------------------------------------------- connection panel
console.log('\nConnection details panel');

const conn = await mount('dnn/components/ConnectionDetails.svelte', {
  targetLayerIndex: 1,
  sourceIndex: 2,
  targetIndex: 0
});
const connText = textOf(conn.target);

check('labels the panel as a connection', connText.includes('Connection'));
check('names the source endpoint', connText.includes('Input'));
check('names the destination endpoint', connText.includes('Hidden Layer 1'));
check('shows both From and To', connText.includes('From') && connText.includes('To'));
check('describes the sign', /positive|negative/.test(connText));
check('shows the contribution section', connText.includes('Contribution'));
check('explains the weight in words', connText.includes('influence'));

const connControls = [];
(function walk(n) { if (n.nodeName === 'input') connControls.push(n); n.childNodes.forEach(walk); })(conn.target);
check('offers a weight slider',
  connControls.some((c) => c.getAttribute('type') === 'range'));
check('offers an exact weight field',
  connControls.some((c) => c.getAttribute('type') === 'number'));

const quickButtons = [];
(function walk(n) { if (n.nodeName === 'button') quickButtons.push(n); n.childNodes.forEach(walk); })(conn.target);
check('offers quick weight adjustments', quickButtons.length >= 6,
  `${quickButtons.length} buttons`);

// The displayed weight must match the network's stored weight.
const storedWeight = net.layers[1].neurons[0].weights[2];
check('displays the real stored weight',
  connText.includes(storedWeight.toFixed(2)) || connText.includes(storedWeight.toFixed(3)),
  `looking for ${storedWeight.toFixed(3)}`);

conn.instance.$destroy();

// ---------------------------------------------------------------- reactivity
console.log('\nPanels react to edits');

const reactive = await mount('dnn/components/ConnectionDetails.svelte', {
  targetLayerIndex: 1,
  sourceIndex: 0,
  targetIndex: 0
});
const before = textOf(reactive.target);

// Edit through the real store action the UI calls.
stores.updateWeight(1, 0, 0, 2.75);
await new Promise((r) => setTimeout(r, 0));
const after = textOf(reactive.target);

check('the panel updates when the weight changes',
  before !== after && after.includes('2.75'),
  `after: ${after.slice(0, 80)}`);
reactive.instance.$destroy();

// A neuron panel must follow an input change.
const following = await mount('dnn/components/NeuronDetails.svelte', {
  layerIndex: 1,
  index: 0
});
const beforeInput = textOf(following.target);
stores.setInput(0, -0.9);
await new Promise((r) => setTimeout(r, 0));
const afterInput = textOf(following.target);
check('the neuron panel updates when an input changes', beforeInput !== afterInput);
following.instance.$destroy();

// ---------------------------------------------------------------- selection
console.log('\nSelection behaviour');
stores.clearSelection();

let selNeuron;
let selConn;
stores.selectedNeuron.subscribe((v) => { selNeuron = v; });
stores.selectedConnection.subscribe((v) => { selConn = v; });

stores.selectNeuron(1, 2);
check('selecting a neuron records it',
  selNeuron !== undefined && selNeuron.layerIndex === 1 && selNeuron.index === 2);

// Selecting a connection must clear the neuron, since the rail shows one panel.
stores.selectConnection(1, 0, 2);
check('selecting a connection clears the neuron selection', selNeuron === undefined);
check('selecting a connection records it', selConn !== undefined);

stores.selectNeuron(2, 1);
check('selecting a neuron clears the connection selection', selConn === undefined);

// Clicking the same neuron again closes the panel.
stores.selectNeuron(2, 1);
check('re-selecting the same neuron deselects it', selNeuron === undefined);

stores.selectConnection(1, 0, 0);
stores.selectConnection(1, 0, 0);
check('re-selecting the same connection deselects it', selConn === undefined);

stores.selectNeuron(1, 1);
stores.clearSelection();
check('clearSelection clears both', selNeuron === undefined && selConn === undefined);

console.log(
  failures === 0
    ? `\nAll ${checks} checks passed.`
    : `\n${failures} of ${checks} checks failed.`
);
process.exit(failures === 0 ? 0 : 1);
