import { mkdirSync, copyFileSync, readdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define paths
const rootDir = join(__dirname, '..', '..');
const sourceDir = join(rootDir, 'case1', 'data');
const targetDir = join(__dirname, '..', 'public', 'case1', 'data');

// Create target directory if it doesn't exist
mkdirSync(targetDir, { recursive: true });

// Copy all .npy files
if (existsSync(sourceDir)) {
  const files = readdirSync(sourceDir).filter(file => file.endsWith('.npy'));
  
  console.log(`Copying ${files.length} data files from case1/data to frontend/public/case1/data`);
  
  files.forEach(file => {
    const sourcePath = join(sourceDir, file);
    const targetPath = join(targetDir, file);
    copyFileSync(sourcePath, targetPath);
    console.log(`  ✓ Copied ${file}`);
  });
  
  console.log('Data files copied successfully!');
} else {
  console.warn(`Warning: Source directory ${sourceDir} does not exist. Run the data generation script first.`);
}
