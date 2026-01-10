import { mkdirSync, copyFileSync, readdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define paths
const rootDir = join(__dirname, '..');
const dataset1Dir = join(rootDir, 'dataset1', 'data');
const case1Dir = join(rootDir, 'case1', 'data');
const case2Dir = join(rootDir, 'case2', 'data');
const case3Dir = join(rootDir, 'case3', 'data');
const case4Dir = join(rootDir, 'case4', 'data');
const targetCase1Dir = join(__dirname, '..', 'public', 'case1', 'data');
const targetCase2Dir = join(__dirname, '..', 'public', 'case2', 'data');
const targetCase3Dir = join(__dirname, '..', 'public', 'case3', 'data');
const targetCase4Dir = join(__dirname, '..', 'public', 'case4', 'data');

// Create target directories if they don't exist
mkdirSync(targetCase1Dir, { recursive: true });
mkdirSync(targetCase2Dir, { recursive: true });
mkdirSync(targetCase3Dir, { recursive: true });
mkdirSync(targetCase4Dir, { recursive: true });

// Copy dataset1 files to both case1 and case2
if (existsSync(dataset1Dir)) {
  const files = readdirSync(dataset1Dir).filter(file => file.endsWith('.npy'));
  
  console.log(`Copying ${files.length} data files from dataset1/data to public/case1/data and case2/data`);
  
  files.forEach(file => {
    const sourcePath = join(dataset1Dir, file);
    // Copy to case1
    copyFileSync(sourcePath, join(targetCase1Dir, file));
    // Copy to case2
    copyFileSync(sourcePath, join(targetCase2Dir, file));
    console.log(`  ✓ Copied ${file}`);
  });
  
  console.log('Dataset1 files copied successfully!');
} else {
  console.warn(`Warning: Source directory ${dataset1Dir} does not exist. Run the data generation script first.`);
}

// Copy case1-specific files
if (existsSync(case1Dir)) {
  const case1Files = readdirSync(case1Dir).filter(file => 
    file.endsWith('.npy') || file.endsWith('.json') || file.endsWith('.csv') || file.endsWith('.png')
  );
  
  console.log(`\nCopying ${case1Files.length} case1-specific files to public/case1/data`);
  
  case1Files.forEach(file => {
    const sourcePath = join(case1Dir, file);
    const targetPath = join(targetCase1Dir, file);
    copyFileSync(sourcePath, targetPath);
    console.log(`  ✓ Copied ${file}`);
  });
  
  console.log('Case1-specific files copied successfully!');
}

// Copy case2-specific files
if (existsSync(case2Dir)) {
  const case2Files = readdirSync(case2Dir).filter(file => 
    file.endsWith('.npy') || file.endsWith('.json') || file.endsWith('.csv') || file.endsWith('.png')
  );
  
  console.log(`\nCopying ${case2Files.length} case2-specific files to public/case2/data`);
  
  case2Files.forEach(file => {
    const sourcePath = join(case2Dir, file);
    const targetPath = join(targetCase2Dir, file);
    copyFileSync(sourcePath, targetPath);
    console.log(`  ✓ Copied ${file}`);
  });
  
  console.log('Case2-specific files copied successfully!');
}

// Copy case3-specific files
if (existsSync(case3Dir)) {
  const case3Files = readdirSync(case3Dir).filter(file => 
    file.endsWith('.npy') || file.endsWith('.json') || file.endsWith('.csv') || file.endsWith('.png')
  );
  
  console.log(`\nCopying ${case3Files.length} case3-specific files to public/case3/data`);
  
  case3Files.forEach(file => {
    const sourcePath = join(case3Dir, file);
    const targetPath = join(targetCase3Dir, file);
    copyFileSync(sourcePath, targetPath);
    console.log(`  ✓ Copied ${file}`);
  });
  
  console.log('Case3-specific files copied successfully!');
}

// Copy case4-specific files
if (existsSync(case4Dir)) {
  const case4Files = readdirSync(case4Dir).filter(file => 
    file.endsWith('.npy') || file.endsWith('.json') || file.endsWith('.csv') || file.endsWith('.png')
  );
  
  console.log(`\nCopying ${case4Files.length} case4-specific files to public/case4/data`);
  
  case4Files.forEach(file => {
    const sourcePath = join(case4Dir, file);
    const targetPath = join(targetCase4Dir, file);
    copyFileSync(sourcePath, targetPath);
    console.log(`  ✓ Copied ${file}`);
  });
  
  console.log('Case4-specific files copied successfully!');
}
