/**
 * Dataset definition for DNN Explainer.
 *
 * The task: classify a 5x5 binary glyph into one of four shape classes.
 * A 25-dimensional input is small enough that EVERY input feature, weight and
 * activation can be drawn on screen at once, which is what makes a dense
 * network worth visualizing.
 *
 * The four base glyphs are hand-authored below. The training set is built by
 * deterministic augmentation (translation + pixel noise) so that the exact same
 * dataset is reproducible from a seed.
 */

const GRID = 5;
const CLASSES = ['L', 'T', 'X', 'O'];

// Hand-authored prototypes. 1 = ink, 0 = background.
const PROTOTYPES = {
  L: [
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]
  ],
  T: [
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
  ],
  X: [
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
  ],
  O: [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
  ]
};

/** Deterministic PRNG so the dataset and the training run are reproducible. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Shift a grid by (dr, dc), filling vacated cells with background. */
function shift(grid, dr, dc) {
  const out = [];
  for (let r = 0; r < GRID; r++) {
    const row = [];
    for (let c = 0; c < GRID; c++) {
      const sr = r - dr;
      const sc = c - dc;
      row.push(sr >= 0 && sr < GRID && sc >= 0 && sc < GRID ? grid[sr][sc] : 0);
    }
    out.push(row);
  }
  return out;
}

/** True when shifting would push ink off the canvas (which destroys the class). */
function shiftIsSafe(grid, dr, dc) {
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      if (grid[r][c] === 1) {
        const tr = r + dr;
        const tc = c + dc;
        if (tr < 0 || tr >= GRID || tc < 0 || tc >= GRID) return false;
      }
    }
  }
  return true;
}

function flatten(grid) {
  const out = [];
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) out.push(grid[r][c]);
  }
  return out;
}

/**
 * Build the augmented dataset.
 * @param {number} seed PRNG seed
 * @param {number} perClass Number of examples to generate per class
 * @param {number} flipProb Probability of flipping any given pixel
 */
function buildDataset(seed = 42, perClass = 400, flipProb = 0.06) {
  const rand = mulberry32(seed);
  const xs = [];
  const ys = [];

  CLASSES.forEach((cls, label) => {
    const proto = PROTOTYPES[cls];

    // Enumerate the legal translations for this glyph once.
    const shifts = [];
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (shiftIsSafe(proto, dr, dc)) shifts.push([dr, dc]);
      }
    }

    for (let n = 0; n < perClass; n++) {
      const [dr, dc] = shifts[Math.floor(rand() * shifts.length)];
      const grid = shift(proto, dr, dc);

      // Pixel noise.
      for (let r = 0; r < GRID; r++) {
        for (let c = 0; c < GRID; c++) {
          if (rand() < flipProb) grid[r][c] = grid[r][c] === 1 ? 0 : 1;
        }
      }

      xs.push(flatten(grid));
      ys.push(label);
    }
  });

  return { xs, ys };
}

module.exports = {
  GRID,
  CLASSES,
  PROTOTYPES,
  buildDataset,
  flatten,
  mulberry32
};
