/**
 * Dataset management for custom user-uploaded training data.
 *
 * Handles parsing, validation, splitting, and normalization of CSV files.
 */

export const DATASET_STATUS = {
  EMPTY: 'empty',
  LOADING: 'loading',
  VALID: 'valid',
  ERROR: 'error'
};

/**
 * Parse CSV text into records.
 * Expects first row to be headers, last column to be the target class.
 *
 * @param {string} csvText
 * @returns {{headers: string[], records: any[], error?: string}}
 */
export const parseCSV = (csvText) => {
  const lines = csvText.trim().split('\n').filter((line) => line.trim().length > 0);
  if (lines.length < 2) return { headers: [], records: [], error: 'CSV must have at least a header and one data row' };

  const [headerLine, ...dataLines] = lines;
  const headers = headerLine.split(',').map((h) => h.trim());

  if (headers.length < 2) {
    return { headers: [], records: [], error: 'CSV must have at least one feature and one target column' };
  }

  const records = [];
  for (let i = 0; i < dataLines.length; i++) {
    const parts = dataLines[i].split(',').map((p) => p.trim());
    if (parts.length !== headers.length) {
      return {
        headers: [],
        records: [],
        error: `Row ${i + 2} has ${parts.length} columns, expected ${headers.length}`
      };
    }
    records.push(parts);
  }

  return { headers, records };
};

/**
 * Validate and convert parsed records into feature/target pairs.
 *
 * @param {string[]} headers
 * @param {string[][]} records
 * @returns {{examples: {inputs: number[], target: number}[], classes: string[], error?: string}}
 */
export const validateDataset = (headers, records) => {
  const featureHeaders = headers.slice(0, -1);
  const targetHeader = headers[headers.length - 1];
  const classSet = new Set();
  const examples = [];

  for (let i = 0; i < records.length; i++) {
    const record = records[i];
    const inputs = record.slice(0, -1);
    const target = record[record.length - 1];

    const numInputs = inputs.map((x) => {
      const num = parseFloat(x);
      if (isNaN(num)) {
        throw new Error(`Row ${i + 2}, feature "${featureHeaders[inputs.indexOf(x)]}": "${x}" is not a number`);
      }
      return num;
    });

    if (target === '' || target === undefined) {
      throw new Error(`Row ${i + 2}: target is empty`);
    }

    classSet.add(target);
    examples.push({ inputs: numInputs, target });
  }

  if (classSet.size < 2) {
    return { examples: [], classes: [], error: `Dataset has only ${classSet.size} class(es); need at least 2` };
  }

  const classes = Array.from(classSet).sort();
  const indexed = examples.map((ex) => ({
    inputs: ex.inputs,
    target: classes.indexOf(ex.target)
  }));

  return { examples: indexed, classes };
};

/**
 * Normalize features to [0, 1] range.
 *
 * @param {object[]} examples
 * @returns {{normalized: object[], stats: {min: number[], max: number[]}}}
 */
export const normalizeDataset = (examples) => {
  if (examples.length === 0) return { normalized: [], stats: { min: [], max: [] } };

  const inputSize = examples[0].inputs.length;
  const min = Array(inputSize).fill(Infinity);
  const max = Array(inputSize).fill(-Infinity);

  for (const ex of examples) {
    for (let i = 0; i < inputSize; i++) {
      if (ex.inputs[i] < min[i]) min[i] = ex.inputs[i];
      if (ex.inputs[i] > max[i]) max[i] = ex.inputs[i];
    }
  }

  const normalized = examples.map((ex) => ({
    inputs: ex.inputs.map((v, i) => {
      const range = max[i] - min[i];
      return range === 0 ? 0.5 : (v - min[i]) / range;
    }),
    target: ex.target
  }));

  return { normalized, stats: { min, max } };
};

/**
 * Split dataset into train and test sets.
 *
 * @param {object[]} examples
 * @param {number} trainRatio (0..1)
 * @param {number} seed
 * @returns {{train: object[], test: object[]}}
 */
export const splitDataset = (examples, trainRatio = 0.8, seed = 123) => {
  const shuffled = [...examples];

  // Fisher-Yates shuffle with seed
  let a = seed;
  const rand = () => {
    a = (a * 9301 + 49297) % 233280;
    return a / 233280;
  };

  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const splitIndex = Math.ceil(shuffled.length * trainRatio);
  return { train: shuffled.slice(0, splitIndex), test: shuffled.slice(splitIndex) };
};

/**
 * Prepare dataset: parse → validate → normalize → split.
 *
 * @param {string} csvText
 * @param {number} trainRatio
 * @returns {{status: string, train?: object[], test?: object[], classes?: string[], error?: string}}
 */
export const prepareDataset = (csvText, trainRatio = 0.8) => {
  try {
    const { headers, records, error: parseError } = parseCSV(csvText);
    if (parseError) return { status: DATASET_STATUS.ERROR, error: parseError };

    const { examples, classes, error: validateError } = validateDataset(headers, records);
    if (validateError) return { status: DATASET_STATUS.ERROR, error: validateError };

    const { normalized, stats } = normalizeDataset(examples);
    const { train, test } = splitDataset(normalized, trainRatio);

    return {
      status: DATASET_STATUS.VALID,
      train,
      test,
      classes,
      stats,
      size: { total: examples.length, train: train.length, test: test.length }
    };
  } catch (err) {
    return { status: DATASET_STATUS.ERROR, error: err.message };
  }
};
