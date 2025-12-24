import { test, expect } from '@playwright/test';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

/**
 * Creates a valid NPY file buffer from a 2D array of data.
 * 
 * @param {number[][]} data - 2D array of numbers to encode
 * @param {number[]} shape - Array shape [rows, cols]
 * @returns {Uint8Array} - Buffer containing valid NPY file data
 */
function createNpyBuffer(data, shape) {
  // NPY format specification: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
  const magic = new Uint8Array([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]); // \x93NUMPY
  const version = new Uint8Array([0x01, 0x00]); // Version 1.0
  
  // Create header dict string
  const dtype = '<f8'; // Little-endian float64
  const shapeStr = `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  const headerDict = `{'descr': '${dtype}', 'fortran_order': False, 'shape': ${shapeStr}, }`;
  
  // Pad header to make total header length (including magic + version + header_len) divisible by 64
  const baseLen = magic.length + version.length + 2 + headerDict.length + 1; // +1 for newline
  const padding = (64 - (baseLen % 64)) % 64;
  const paddedHeader = headerDict + ' '.repeat(padding) + '\n';
  
  // Header length as little-endian uint16
  const headerLen = paddedHeader.length;
  const headerLenBytes = new Uint8Array([headerLen & 0xFF, (headerLen >> 8) & 0xFF]);
  
  // Create data buffer (float64)
  const flatData = data.flat(Infinity);
  const dataBuffer = new Float64Array(flatData);
  
  // Combine all parts
  const totalLen = magic.length + version.length + headerLenBytes.length + paddedHeader.length + dataBuffer.byteLength;
  const buffer = new Uint8Array(totalLen);
  
  let offset = 0;
  buffer.set(magic, offset); offset += magic.length;
  buffer.set(version, offset); offset += version.length;
  buffer.set(headerLenBytes, offset); offset += headerLenBytes.length;
  buffer.set(new TextEncoder().encode(paddedHeader), offset); offset += paddedHeader.length;
  buffer.set(new Uint8Array(dataBuffer.buffer), offset);
  
  return buffer;
}

// Generate random samples for testing
function generateTestPredictions(nTestPoints, nSamplesPerPoint) {
  const data = [];
  for (let i = 0; i < nTestPoints; i++) {
    const row = [];
    // Generate samples around some base value with noise
    const baseValue = (i / nTestPoints) * 10 - 5;
    for (let j = 0; j < nSamplesPerPoint; j++) {
      row.push(baseValue + (Math.random() - 0.5) * 4);
    }
    data.push(row);
  }
  return data;
}

test.describe('Case2 Upload and Scoring', () => {
  test.beforeAll(async () => {
    // Create test fixtures directory if needed
    const fixturesDir = join(process.cwd(), 'tests', 'fixtures');
    if (!existsSync(fixturesDir)) {
      mkdirSync(fixturesDir, { recursive: true });
    }
  });

  test('page loads correctly', async ({ page }) => {
    await page.goto('/case2');
    
    // Check page title
    await expect(page.locator('h1')).toContainText('Case Study 2: Distribution Sampling');
    
    // Check that energy score explanation exists (use exact match)
    await expect(page.getByText('Energy Score', { exact: true })).toBeVisible();
    
    // Check for the goal message
    await expect(page.getByText('Try to achieve an energy score less than 2.1')).toBeVisible();
    
    // Check for upload section
    await expect(page.getByText('Submit Your Predictions')).toBeVisible();
    
    // Check that file input exists
    await expect(page.locator('input[type="file"]')).toBeVisible();
  });

  test('shows error for wrong number of test points', async ({ page }) => {
    await page.goto('/case2');
    
    // Create a file with wrong number of test points (50 instead of 100)
    const fixturesDir = join(process.cwd(), 'tests', 'fixtures');
    const wrongPointsData = generateTestPredictions(50, 10);
    const wrongPointsBuffer = createNpyBuffer(wrongPointsData, [50, 10]);
    const wrongPointsPath = join(fixturesDir, 'wrong_points.npy');
    writeFileSync(wrongPointsPath, wrongPointsBuffer);
    
    // Upload the file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(wrongPointsPath);
    
    // Click calculate
    await page.getByRole('button', { name: 'Calculate Score' }).click();
    
    // Check for error message
    await expect(page.getByText(/Expected 100 test points/)).toBeVisible();
  });

  test('shows error for too few samples', async ({ page }) => {
    await page.goto('/case2');
    
    // Create a file with too few samples (3 instead of 5-100)
    const fixturesDir = join(process.cwd(), 'tests', 'fixtures');
    const tooFewData = generateTestPredictions(100, 3);
    const tooFewBuffer = createNpyBuffer(tooFewData, [100, 3]);
    const tooFewPath = join(fixturesDir, 'too_few_samples.npy');
    writeFileSync(tooFewPath, tooFewBuffer);
    
    // Upload the file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(tooFewPath);
    
    // Click calculate
    await page.getByRole('button', { name: 'Calculate Score' }).click();
    
    // Check for error message
    await expect(page.getByText(/Expected between 5 and 100 samples/)).toBeVisible();
  });

  test('shows error for too many samples', async ({ page }) => {
    await page.goto('/case2');
    
    // Create a file with too many samples (150 instead of 5-100)
    const fixturesDir = join(process.cwd(), 'tests', 'fixtures');
    const tooManyData = generateTestPredictions(100, 150);
    const tooManyBuffer = createNpyBuffer(tooManyData, [100, 150]);
    const tooManyPath = join(fixturesDir, 'too_many_samples.npy');
    writeFileSync(tooManyPath, tooManyBuffer);
    
    // Upload the file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(tooManyPath);
    
    // Click calculate
    await page.getByRole('button', { name: 'Calculate Score' }).click();
    
    // Check for error message
    await expect(page.getByText(/Expected between 5 and 100 samples/)).toBeVisible();
  });

  test('calculates score with minimum samples (5)', async ({ page }) => {
    await page.goto('/case2');
    
    // Create a valid file with minimum samples
    const fixturesDir = join(process.cwd(), 'tests', 'fixtures');
    const minData = generateTestPredictions(100, 5);
    const minBuffer = createNpyBuffer(minData, [100, 5]);
    const minPath = join(fixturesDir, 'min_samples.npy');
    writeFileSync(minPath, minBuffer);
    
    // Upload the file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(minPath);
    
    // Click calculate
    await page.getByRole('button', { name: 'Calculate Score' }).click();
    
    // Check for score display
    await expect(page.getByText(/Energy Score =/)).toBeVisible();
    await expect(page.getByText('(using 5 samples per test point)')).toBeVisible();
  });

  test('calculates score with maximum samples (100)', async ({ page }) => {
    await page.goto('/case2');
    
    // Create a valid file with maximum samples
    const fixturesDir = join(process.cwd(), 'tests', 'fixtures');
    const maxData = generateTestPredictions(100, 100);
    const maxBuffer = createNpyBuffer(maxData, [100, 100]);
    const maxPath = join(fixturesDir, 'max_samples.npy');
    writeFileSync(maxPath, maxBuffer);
    
    // Upload the file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(maxPath);
    
    // Click calculate
    await page.getByRole('button', { name: 'Calculate Score' }).click();
    
    // Check for score display
    await expect(page.getByText(/Energy Score =/)).toBeVisible();
    await expect(page.getByText('(using 100 samples per test point)')).toBeVisible();
  });

  test('calculates score with typical sample count (20)', async ({ page }) => {
    await page.goto('/case2');
    
    // Create a valid file with typical sample count
    const fixturesDir = join(process.cwd(), 'tests', 'fixtures');
    const typicalData = generateTestPredictions(100, 20);
    const typicalBuffer = createNpyBuffer(typicalData, [100, 20]);
    const typicalPath = join(fixturesDir, 'typical_samples.npy');
    writeFileSync(typicalPath, typicalBuffer);
    
    // Upload the file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(typicalPath);
    
    // Click calculate
    await page.getByRole('button', { name: 'Calculate Score' }).click();
    
    // Check for score display
    await expect(page.getByText(/Energy Score =/)).toBeVisible();
    await expect(page.getByText('(using 20 samples per test point)')).toBeVisible();
    
    // Score should be a reasonable number (positive)
    const scoreText = await page.getByText(/Energy Score =/).textContent();
    const match = scoreText.match(/\d+(?:\.\d+)?/);
    const score = match ? parseFloat(match[0]) : NaN;
    expect(score).toBeGreaterThan(0);
  });

  test('shows appropriate message based on score threshold', async ({ page }) => {
    await page.goto('/case2');
    
    // Create a file (scores will vary, so we just check that the message logic works)
    const fixturesDir = join(process.cwd(), 'tests', 'fixtures');
    const testData = generateTestPredictions(100, 50);
    const testBuffer = createNpyBuffer(testData, [100, 50]);
    const testPath = join(fixturesDir, 'threshold_test.npy');
    writeFileSync(testPath, testBuffer);
    
    // Upload the file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testPath);
    
    // Click calculate
    await page.getByRole('button', { name: 'Calculate Score' }).click();
    
    // Wait for score to appear
    await expect(page.getByText(/Energy Score =/)).toBeVisible();
    
    // Check that one of the threshold messages is visible
    const successMsg = page.getByText('Great job! You beat the target of 2.1!');
    const tryAgainMsg = page.getByText('Try to get below 2.1! Lower is better.');
    
    // One of these should be visible (check with short timeout)
    const [isSuccess, isTryAgain] = await Promise.all([
      successMsg.isVisible({ timeout: 1000 }).catch(() => false),
      tryAgainMsg.isVisible({ timeout: 1000 }).catch(() => false)
    ]);
    expect(isSuccess || isTryAgain).toBeTruthy();
  });
});
