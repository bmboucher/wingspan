#!/usr/bin/env node
/**
 * Architecture analysis script for wingspan codebase.
 * Computes structural patterns from import graph and file paths.
 */

const fs = require('fs');
const path = require('path');

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
  process.exit(1);
}

let input;
try {
  input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (err) {
  console.error('Failed to read input:', err.message);
  process.exit(1);
}

const { fileNodes, importEdges, allEdges } = input;

// ─── A. Directory Grouping ────────────────────────────────────────────────────

// Find common prefix among all file paths
function findCommonPrefix(paths) {
  if (!paths.length) return '';
  const segments = paths.map(p => p.split('/'));
  let prefix = [];
  for (let i = 0; i < segments[0].length; i++) {
    const seg = segments[0][i];
    if (segments.every(s => s[i] === seg)) {
      prefix.push(seg);
    } else {
      break;
    }
  }
  return prefix.join('/');
}

const allFilePaths = fileNodes
  .filter(n => n.filePath)
  .map(n => n.filePath);

const commonPrefix = findCommonPrefix(allFilePaths);
// Common prefix is likely empty or 'src' - let's find the meaningful prefix
// For wingspan, files span: src/wingspan/*, docs/*, tests/*, deploy/*, scripts/*, root files
// The actual prefix for source code is src/wingspan
const srcPrefix = 'src/wingspan/';

function getDirectoryGroup(filePath) {
  if (filePath.startsWith(srcPrefix)) {
    const remainder = filePath.slice(srcPrefix.length);
    const parts = remainder.split('/');
    // If it's a direct file in src/wingspan/, use a "root" group
    if (parts.length === 1) return 'wingspan-root';
    return parts[0];
  }
  // Non-src files: group by top-level directory
  const parts = filePath.split('/');
  if (parts.length === 1) return 'project-root';
  // Special cases
  if (parts[0] === 'docs') return 'docs';
  if (parts[0] === 'tests') return 'tests';
  if (parts[0] === 'deploy') return 'deploy';
  if (parts[0] === 'scripts') return 'scripts';
  if (parts[0] === 'src') return 'src-other';
  return parts[0];
}

const directoryGroups = {};
for (const node of fileNodes) {
  if (!node.filePath) continue;
  const group = getDirectoryGroup(node.filePath);
  if (!directoryGroups[group]) directoryGroups[group] = [];
  directoryGroups[group].push(node.id);
}

// ─── B. Node Type Grouping ────────────────────────────────────────────────────

const nodeTypeGroups = {};
for (const node of fileNodes) {
  const t = node.type || 'file';
  if (!nodeTypeGroups[t]) nodeTypeGroups[t] = [];
  nodeTypeGroups[t].push(node.id);
}

// ─── C. Import Adjacency + Fan-in/Fan-out ────────────────────────────────────

const fanIn = {};
const fanOut = {};
const nodeIdSet = new Set(fileNodes.map(n => n.id));

for (const edge of importEdges) {
  if (edge.type !== 'imports') continue;
  if (!nodeIdSet.has(edge.source) || !nodeIdSet.has(edge.target)) continue;
  fanOut[edge.source] = (fanOut[edge.source] || 0) + 1;
  fanIn[edge.target] = (fanIn[edge.target] || 0) + 1;
}

// ─── D. Cross-Category Dependency Analysis ───────────────────────────────────

const crossCategoryMap = {};
for (const edge of allEdges) {
  // Find source and target node types
  const srcNode = fileNodes.find(n => n.id === edge.source);
  const tgtNode = fileNodes.find(n => n.id === edge.target);
  if (!srcNode || !tgtNode) continue;
  if (srcNode.type === tgtNode.type) continue; // same type
  const key = `${srcNode.type}->${tgtNode.type}:${edge.type}`;
  crossCategoryMap[key] = (crossCategoryMap[key] || 0) + 1;
}

const crossCategoryEdges = Object.entries(crossCategoryMap).map(([key, count]) => {
  const [typePair, edgeType] = key.split(':');
  const [fromType, toType] = typePair.split('->');
  return { fromType, toType, edgeType, count };
});

// ─── E. Inter-Group Import Frequency ─────────────────────────────────────────

const nodeToGroup = {};
for (const node of fileNodes) {
  if (node.filePath) nodeToGroup[node.id] = getDirectoryGroup(node.filePath);
}

const interGroupMap = {};
for (const edge of importEdges) {
  if (edge.type !== 'imports') continue;
  const fromGroup = nodeToGroup[edge.source];
  const toGroup = nodeToGroup[edge.target];
  if (!fromGroup || !toGroup || fromGroup === toGroup) continue;
  const key = `${fromGroup}->${toGroup}`;
  interGroupMap[key] = (interGroupMap[key] || 0) + 1;
}

const interGroupImports = Object.entries(interGroupMap).map(([key, count]) => {
  const [from, to] = key.split('->');
  return { from, to, count };
}).sort((a, b) => b.count - a.count);

// ─── F. Intra-Group Import Density ───────────────────────────────────────────

const intraGroupDensity = {};
const totalEdgesPerGroup = {};

for (const edge of importEdges) {
  if (edge.type !== 'imports') continue;
  const fromGroup = nodeToGroup[edge.source];
  const toGroup = nodeToGroup[edge.target];
  if (!fromGroup || !toGroup) continue;
  totalEdgesPerGroup[fromGroup] = (totalEdgesPerGroup[fromGroup] || 0) + 1;
  if (fromGroup === toGroup) {
    if (!intraGroupDensity[fromGroup]) intraGroupDensity[fromGroup] = { internalEdges: 0, totalEdges: 0, density: 0 };
    intraGroupDensity[fromGroup].internalEdges++;
  }
}

for (const group of Object.keys(directoryGroups)) {
  const total = totalEdgesPerGroup[group] || 0;
  const internal = intraGroupDensity[group]?.internalEdges || 0;
  intraGroupDensity[group] = {
    internalEdges: internal,
    totalEdges: total,
    density: total > 0 ? internal / total : 0
  };
}

// ─── G. Directory Pattern Matching ───────────────────────────────────────────

const DIR_PATTERNS = {
  'routes': 'api', 'api': 'api', 'controllers': 'api', 'endpoints': 'api', 'handlers': 'api',
  'services': 'service', 'core': 'service', 'lib': 'service', 'domain': 'service', 'logic': 'service',
  'models': 'data', 'db': 'data', 'data': 'data', 'persistence': 'data', 'repository': 'data', 'entities': 'data',
  'components': 'ui', 'views': 'ui', 'pages': 'ui', 'ui': 'ui', 'layouts': 'ui', 'screens': 'ui',
  'middleware': 'middleware', 'plugins': 'middleware', 'interceptors': 'middleware', 'guards': 'middleware',
  'utils': 'utility', 'helpers': 'utility', 'common': 'utility', 'shared': 'utility', 'tools': 'utility',
  'config': 'config', 'constants': 'config', 'env': 'config', 'settings': 'config',
  '__tests__': 'test', 'test': 'test', 'tests': 'test', 'spec': 'test', 'specs': 'test',
  'types': 'types', 'interfaces': 'types', 'schemas': 'types', 'contracts': 'types', 'dtos': 'types',
  'hooks': 'hooks',
  'store': 'state', 'state': 'state', 'reducers': 'state', 'actions': 'state', 'slices': 'state',
  'assets': 'assets', 'static': 'assets', 'public': 'assets',
  'migrations': 'data',
  'management': 'config', 'commands': 'config',
  'templatetags': 'utility',
  'signals': 'service',
  'serializers': 'api',
  'cmd': 'entry',
  'internal': 'service',
  'pkg': 'utility',
  'dto': 'types', 'request': 'types', 'response': 'types',
  'entity': 'data',
  'controller': 'api',
  'routers': 'api',
  'composables': 'service',
  'blueprints': 'api',
  'mailers': 'service', 'jobs': 'service', 'channels': 'service',
  'bin': 'entry',
  'docs': 'documentation',
  'deploy': 'infrastructure',
  'scripts': 'utility',
  // wingspan-specific
  'agents': 'service',
  'cards': 'data',
  'compat': 'utility',
  'cloud': 'infrastructure',
  'encode': 'utility',
  'engine': 'service',
  'instrumentation': 'service',
  'model': 'utility',
  'players': 'service',
  'reporting': 'utility',
  'setup_model': 'utility',
  'tournament': 'service',
  'training': 'service',
  'powers': 'service',
  'wingspan-root': 'entry',
  'project-root': 'entry',
};

const patternMatches = {};
for (const group of Object.keys(directoryGroups)) {
  patternMatches[group] = DIR_PATTERNS[group] || 'unknown';
}

// ─── H. Deployment Topology Detection ────────────────────────────────────────

const allFilePaths2 = fileNodes.map(n => n.filePath || '');
const deploymentTopology = {
  hasDockerfile: allFilePaths2.some(p => p === 'Dockerfile' || p.includes('/Dockerfile')),
  hasCompose: allFilePaths2.some(p => p.includes('docker-compose')),
  hasK8s: allFilePaths2.some(p => p.includes('k8s') || p.includes('kubernetes') || p.endsWith('.yaml') && p.includes('deploy')),
  hasTerraform: allFilePaths2.some(p => p.endsWith('.tf') || p.endsWith('.tfvars')),
  hasCI: allFilePaths2.some(p => p.includes('.github') || p.includes('.gitlab') || p.includes('Jenkinsfile')),
  infraFiles: fileNodes
    .filter(n => {
      const fp = n.filePath || '';
      return fp === 'Dockerfile' || fp.includes('docker-compose') || fp.endsWith('.tf') ||
             fp.includes('.github/workflows') || fp.includes('deploy/');
    })
    .map(n => n.filePath)
};

// ─── I. Data Pipeline Detection ──────────────────────────────────────────────

const dataPipeline = {
  schemaFiles: fileNodes.filter(n => (n.filePath || '').endsWith('.graphql') || (n.filePath || '').endsWith('.proto') || (n.tags || []).includes('schema')).map(n => n.filePath),
  migrationFiles: fileNodes.filter(n => (n.filePath || '').includes('migration') || (n.filePath || '').endsWith('.sql')).map(n => n.filePath),
  dataModelFiles: fileNodes.filter(n => (n.tags || []).includes('data-model')).map(n => n.filePath),
  apiHandlerFiles: fileNodes.filter(n => (n.tags || []).includes('api-handler')).map(n => n.filePath)
};

// ─── J. Documentation Coverage ───────────────────────────────────────────────

const groupsWithDocs = new Set();
for (const node of fileNodes) {
  if (node.type === 'document') {
    const group = getDirectoryGroup(node.filePath || '');
    groupsWithDocs.add(group);
  }
}
const totalGroups = Object.keys(directoryGroups).length;
const undocumentedGroups = Object.keys(directoryGroups).filter(g => !groupsWithDocs.has(g));

const docCoverage = {
  groupsWithDocs: groupsWithDocs.size,
  totalGroups,
  coverageRatio: groupsWithDocs.size / totalGroups,
  undocumentedGroups
};

// ─── K. Dependency Direction ──────────────────────────────────────────────────

const dependencyDirection = interGroupImports
  .filter(e => e.count >= 2)
  .map(e => ({ dependent: e.from, dependsOn: e.to }));

// ─── File Stats ───────────────────────────────────────────────────────────────

const filesPerGroup = {};
for (const [group, ids] of Object.entries(directoryGroups)) {
  filesPerGroup[group] = ids.length;
}

const nodeTypeCounts = {};
for (const [type, ids] of Object.entries(nodeTypeGroups)) {
  nodeTypeCounts[type] = ids.length;
}

const fileStats = {
  totalFileNodes: fileNodes.length,
  filesPerGroup,
  nodeTypeCounts
};

// ─── Build Output ─────────────────────────────────────────────────────────────

const output = {
  scriptCompleted: true,
  directoryGroups,
  nodeTypeGroups,
  crossCategoryEdges,
  interGroupImports,
  intraGroupDensity,
  patternMatches,
  deploymentTopology,
  dataPipeline,
  docCoverage,
  dependencyDirection,
  fileStats,
  fileFanIn: fanIn,
  fileFanOut: fanOut
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`Analysis complete. Total nodes: ${fileNodes.length}`);
  console.log(`Directory groups: ${Object.keys(directoryGroups).join(', ')}`);
  process.exit(0);
} catch (err) {
  console.error('Failed to write output:', err.message);
  process.exit(1);
}
