/**
 * Phase 7a regression suite: dataset parsing, validation, and preparation.
 */

import {
  parseCSV, validateDataset, normalizeDataset, splitDataset,
  prepareDataset, DATASET_STATUS
} from '../src/dnn/engine/dataset.js';

let passed = 0;
let failed = 0;

const check = (label, condition) => {
  if (condition) {
    passed++;
  } else {
    failed++;
    console.error(`✗ ${label}`);
  }
};

const approx = (a, b, tol = 1e-6) => Math.abs(a - b) < tol;

console.log('\n=== Phase 7a: Dataset Module ===\n');

// ------------------------------------------------------------------ CSV parsing

const validCSV = `feature1,feature2,feature3,class
0.5,0.8,0.2,A
0.3,0.6,0.9,B
0.7,0.4,0.1,A
0.1,0.9,0.5,B`;

const { headers, records, error: parseError } = parseCSV(validCSV);
check('parseCSV extracts headers', headers.length === 4);
check('parseCSV header names correct', headers[0] === 'feature1' && headers[3] === 'class');
check('parseCSV extracts records', records.length === 4);
check('parseCSV no error on valid input', parseError === undefined);
check('parseCSV record structure', records[0].length === 4);

const emptyCSV = parseCSV('header\n');
check('parseCSV rejects empty data', emptyCSV.error !== undefined);

const singleColumnCSV = parseCSV('onlyOneColumn\n1\n2\n3');
check('parseCSV rejects single column', singleColumnCSV.error !== undefined);

const mismatchedCSV = parseCSV('a,b,c\n1,2,3\n4,5\n6,7,8');
check('parseCSV detects column mismatch', mismatchedCSV.error !== undefined);
check('parseCSV error mentions row', mismatchedCSV.error.includes('Row'));

// ------------------------------------------------------------------ validation

const { examples, classes, error: validateError } = validateDataset(headers, records);
check('validateDataset produces examples', examples.length === 4);
check('validateDataset extracts classes', classes.length === 2);
check('validateDataset sorts classes', classes[0] === 'A' && classes[1] === 'B');
check('validateDataset no error', validateError === undefined);
check('validateDataset numeric inputs', examples[0].inputs.length === 3);
check('validateDataset converts first input', approx(examples[0].inputs[0], 0.5));
check('validateDataset target is index', examples[0].target === 0);
check('validateDataset second target correct', examples[1].target === 1);

const nonNumericHeaders = ['x', 'y', 'class'];
const nonNumericRecords = [['1.0', 'not-a-number', 'A']];
try {
  validateDataset(nonNumericHeaders, nonNumericRecords);
  check('validateDataset rejects non-numeric', false);
} catch (err) {
  check('validateDataset rejects non-numeric', err.message.includes('not a number'));
}

const singleClassHeaders = ['a', 'b', 'class'];
const singleClassRecords = [['1', '2', 'X'], ['3', '4', 'X']];
const singleClassResult = validateDataset(singleClassHeaders, singleClassRecords);
check('validateDataset rejects single class', singleClassResult.error !== undefined);
check('validateDataset single-class error message', singleClassResult.error.includes('only 1'));

const emptyTargetHeaders = ['a', 'b', 'class'];
const emptyTargetRecords = [['1', '2', '']];
try {
  validateDataset(emptyTargetHeaders, emptyTargetRecords);
  check('validateDataset rejects empty target', false);
} catch (err) {
  check('validateDataset rejects empty target', err.message.includes('empty'));
}

// ------------------------------------------------------------------ normalization

const rawExamples = [
  { inputs: [0, 10], target: 0 },
  { inputs: [5, 20], target: 1 },
  { inputs: [10, 30], target: 0 }
];

const { normalized, stats } = normalizeDataset(rawExamples);
check('normalizeDataset returns normalized', normalized.length === 3);
check('normalizeDataset min tracked', stats.min.length === 2);
check('normalizeDataset max tracked', stats.max.length === 2);
check('normalizeDataset min correct', stats.min[0] === 0 && stats.min[1] === 10);
check('normalizeDataset max correct', stats.max[0] === 10 && stats.max[1] === 30);
check('normalizeDataset first input normalized', approx(normalized[0].inputs[0], 0));
check('normalizeDataset last input normalized', approx(normalized[2].inputs[0], 1));
check('normalizeDataset middle value', approx(normalized[1].inputs[0], 0.5));
check('normalizeDataset targets preserved', normalized[1].target === 1);

const constantExamples = [
  { inputs: [5, 5], target: 0 },
  { inputs: [5, 5], target: 1 }
];
const constantNorm = normalizeDataset(constantExamples);
check('normalizeDataset constant feature handled', approx(constantNorm.normalized[0].inputs[0], 0.5));

const emptyNorm = normalizeDataset([]);
check('normalizeDataset empty array', emptyNorm.normalized.length === 0);

// ------------------------------------------------------------------ splitting

const splitExamples = Array.from({ length: 10 }, (_, i) => ({ inputs: [i], target: i % 2 }));
const { train, test } = splitDataset(splitExamples, 0.8, 42);
check('splitDataset train size', train.length === 8);
check('splitDataset test size', test.length === 2);
check('splitDataset total preserved', train.length + test.length === 10);

const split2 = splitDataset(splitExamples, 0.8, 42);
check('splitDataset deterministic', train[0].inputs[0] === split2.train[0].inputs[0]);
check('splitDataset deterministic order', train[3].inputs[0] === split2.train[3].inputs[0]);

const split3 = splitDataset(splitExamples, 0.8, 999);
const orderChanged = JSON.stringify(train.map(e => e.inputs[0])) !== JSON.stringify(split3.train.map(e => e.inputs[0]));
check('splitDataset different seed changes order', orderChanged);

const smallSplit = splitDataset([{ inputs: [1], target: 0 }], 0.8);
check('splitDataset single example in train', smallSplit.train.length === 1);
check('splitDataset single example test empty', smallSplit.test.length === 0);

// ------------------------------------------------------------------ full pipeline

const irisSubset = `sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
4.9,3.0,1.4,0.2,setosa
6.4,3.2,4.5,1.5,versicolor
5.8,2.7,5.1,1.9,virginica`;

const result = prepareDataset(irisSubset, 0.7);
check('prepareDataset status valid', result.status === DATASET_STATUS.VALID);
check('prepareDataset has train', result.train.length > 0);
check('prepareDataset has test', result.test.length > 0);
check('prepareDataset classes extracted', result.classes.length === 3);
check('prepareDataset classes sorted', result.classes[0] === 'setosa');
check('prepareDataset size info', result.size.total === 6);
check('prepareDataset train size info', result.size.train === 5);
check('prepareDataset test size info', result.size.test === 1);
check('prepareDataset stats tracked', result.stats.min.length === 4);
check('prepareDataset train inputs normalized', result.train[0].inputs[0] >= 0 && result.train[0].inputs[0] <= 1);

const errorResult = prepareDataset('not,enough\n1,2');
check('prepareDataset error status', errorResult.status === DATASET_STATUS.ERROR);
check('prepareDataset error message', errorResult.error !== undefined);
check('prepareDataset no train on error', errorResult.train === undefined);

const singleClassCSV = `a,b,class
1,2,X
3,4,X`;
const singleResult = prepareDataset(singleClassCSV);
check('prepareDataset single class error', singleResult.status === DATASET_STATUS.ERROR);

// ------------------------------------------------------------------ edge cases

const trailingComma = parseCSV('a,b,class,\n1,2,X,');
check('parseCSV handles trailing comma', trailingComma.headers.length === 4);

const windowsLineEndings = parseCSV('a,b,c\r\n1,2,3\r\n4,5,6');
check('parseCSV handles windows line endings', windowsLineEndings.records.length === 2);

const extraWhitespace = parseCSV('  a  , b , c  \n 1 , 2 , 3 ');
check('parseCSV trims whitespace', extraWhitespace.headers[0] === 'a');
check('parseCSV trims values', extraWhitespace.records[0][0] === '1');

// ------------------------------------------------------------------ summary

console.log(`\n${passed} checks passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
