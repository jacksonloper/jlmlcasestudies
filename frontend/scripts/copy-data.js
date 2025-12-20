import { mkdirSync, copyFileSync, readdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define paths
const rootDir = join(__dirname, '..', '..');
const sourceDir = join(rootDir, 'dataset1', 'data');
const targetDirs = [
  join(__dirname, '..', 'public', 'case1', 'data'),
  join(__dirname, '..', 'public', 'case2', 'data')
];

// Create target directories if they don't exist
targetDirs.forEach(dir => mkdirSync(dir, { recursive: true }));

// Copy all .npy files to both case1 and case2
if (existsSync(sourceDir)) {
  const files = readdirSync(sourceDir).filter(file => file.endsWith('.npy'));
  
  console.log(`Copying ${files.length} data files from dataset1/data to frontend/public/case1/data and case2/data`);
  
  files.forEach(file => {
    const sourcePath = join(sourceDir, file);
    targetDirs.forEach(targetDir => {
      const targetPath = join(targetDir, file);
      copyFileSync(sourcePath, targetPath);
    });
    console.log(`  ✓ Copied ${file}`);
  });
  
  console.log('Data files copied successfully!');
} else {
  console.warn(`Warning: Source directory ${sourceDir} does not exist. Run the data generation script first.`);
}
