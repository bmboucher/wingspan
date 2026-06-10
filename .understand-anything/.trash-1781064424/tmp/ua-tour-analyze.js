#!/usr/bin/env node
/**
 * Graph topology analysis script for tour design.
 * Usage: node ua-tour-analyze.js <input.json> <output.json>
 */

const fs = require('fs');

// --- I/O setup ---
const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  process.stderr.write('Usage: node ua-tour-analyze.js <input.json> <output.json>\n');
  process.exit(1);
}

let inputData;
try {
  inputData = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (err) {
  process.stderr.write(`Failed to read/parse input: ${err.message}\n`);
  process.exit(1);
}

const { nodes, edges, layers } = inputData;

// --- A. Fan-In Ranking ---
const fanInMap = {};
const fanOutMap = {};
for (const node of nodes) {
  fanInMap[node.id] = 0;
  fanOutMap[node.id] = 0;
}
for (const edge of edges) {
  if (fanInMap[edge.target] !== undefined) fanInMap[edge.target]++;
  if (fanOutMap[edge.source] !== undefined) fanOutMap[edge.source]++;
}

const nodeById = {};
for (const node of nodes) nodeById[node.id] = node;

const fanInRanking = Object.entries(fanInMap)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 20)
  .map(([id, fanIn]) => ({ id, fanIn, name: nodeById[id]?.name || id }));

// --- B. Fan-Out Ranking ---
const fanOutRanking = Object.entries(fanOutMap)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 20)
  .map(([id, fanOut]) => ({ id, fanOut, name: nodeById[id]?.name || id }));

// --- C. Entry Point Candidates ---
const ENTRY_NAMES = new Set([
  'index.ts','index.js','main.ts','main.js','app.ts','app.js',
  'server.ts','server.js','mod.rs','main.go','main.py','main.rs',
  'manage.py','app.py','wsgi.py','asgi.py','run.py','__main__.py',
  'Application.java','Main.java','Program.cs','config.ru','index.php',
  'App.swift','Application.kt','main.cpp','main.c'
]);

const totalNodes = nodes.length;
const fanOutValues = Object.values(fanOutMap).sort((a, b) => a - b);
const fanInValues = Object.values(fanInMap).sort((a, b) => a - b);
const top10FanOutThreshold = fanOutValues[Math.floor(totalNodes * 0.9)] || 0;
const bottom25FanInThreshold = fanInValues[Math.floor(totalNodes * 0.25)] || 0;

const entryScores = [];
for (const node of nodes) {
  let score = 0;
  if (node.type === 'document') {
    if (node.name === 'README.md' && node.filePath === 'README.md') score += 5;
    else if (node.name.endsWith('.md') && !node.filePath.includes('/')) score += 2;
  } else if (node.type === 'file') {
    if (ENTRY_NAMES.has(node.name)) score += 3;
    const depth = (node.filePath.match(/\//g) || []).length;
    if (depth <= 1) score += 1;
    if ((fanOutMap[node.id] || 0) >= top10FanOutThreshold) score += 1;
    if ((fanInMap[node.id] || 0) <= bottom25FanInThreshold) score += 1;
  }
  if (score > 0) {
    entryScores.push({ id: node.id, score, name: node.name, type: node.type, summary: node.summary || '' });
  }
}
entryScores.sort((a, b) => b.score - a.score);
const entryPointCandidates = entryScores.slice(0, 5);

// --- D. BFS from top code entry point ---
// Skip documentation nodes; find top code entry point
const topCodeEntry = entryScores.find(c => c.type === 'file');

const bfsResult = { startNode: null, order: [], depthMap: {}, byDepth: {} };

if (topCodeEntry) {
  bfsResult.startNode = topCodeEntry.id;
  const visited = new Set();
  const queue = [{ id: topCodeEntry.id, depth: 0 }];
  visited.add(topCodeEntry.id);

  while (queue.length > 0) {
    const { id: currentId, depth } = queue.shift();
    bfsResult.order.push(currentId);
    bfsResult.depthMap[currentId] = depth;
    if (!bfsResult.byDepth[depth]) bfsResult.byDepth[depth] = [];
    bfsResult.byDepth[depth].push(currentId);

    // Follow imports and calls edges (forward direction)
    for (const edge of edges) {
      if (edge.source === currentId && (edge.type === 'imports' || edge.type === 'calls')) {
        if (!visited.has(edge.target) && nodeById[edge.target]) {
          visited.add(edge.target);
          queue.push({ id: edge.target, depth: depth + 1 });
        }
      }
    }
  }
}

// --- E. Non-Code File Inventory ---
const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
for (const node of nodes) {
  if (node.type === 'document') {
    nonCodeFiles.documentation.push({ id: node.id, name: node.name, type: node.type, summary: node.summary || '' });
  } else if (['service', 'pipeline', 'resource'].includes(node.type)) {
    nonCodeFiles.infrastructure.push({ id: node.id, name: node.name, type: node.type, summary: node.summary || '' });
  } else if (['table', 'schema', 'endpoint'].includes(node.type)) {
    nonCodeFiles.data.push({ id: node.id, name: node.name, type: node.type, summary: node.summary || '' });
  } else if (node.type === 'config') {
    nonCodeFiles.config.push({ id: node.id, name: node.name, type: node.type, summary: node.summary || '' });
  }
}

// --- F. Tightly Coupled Clusters ---
// Build adjacency for bidirectional edges
const pairEdgeCounts = {};
for (const edge of edges) {
  const key = [edge.source, edge.target].sort().join('|||');
  if (!pairEdgeCounts[key]) pairEdgeCounts[key] = 0;
  pairEdgeCounts[key]++;
}

// Find bidirectional pairs
const biPairs = [];
for (const edge of edges) {
  const reverseExists = edges.some(e => e.source === edge.target && e.target === edge.source);
  if (reverseExists && edge.source < edge.target) {
    biPairs.push([edge.source, edge.target]);
  }
}

// Build initial clusters from bidirectional pairs
const clusterSets = biPairs.map(pair => new Set(pair));

// Expand clusters: add nodes that connect to 2+ existing cluster members
for (const node of nodes) {
  for (const cluster of clusterSets) {
    if (cluster.has(node.id)) continue;
    let connectionCount = 0;
    for (const memberId of cluster) {
      const connected = edges.some(e =>
        (e.source === node.id && e.target === memberId) ||
        (e.source === memberId && e.target === node.id)
      );
      if (connected) connectionCount++;
    }
    if (connectionCount >= 2) cluster.add(node.id);
  }
}

// Count edges within each cluster and deduplicate
const clusterResults = [];
const seen = new Set();
for (const cluster of clusterSets) {
  const members = Array.from(cluster);
  const key = members.sort().join('|');
  if (seen.has(key)) continue;
  seen.add(key);
  let edgeCount = 0;
  for (const edge of edges) {
    if (cluster.has(edge.source) && cluster.has(edge.target)) edgeCount++;
  }
  clusterResults.push({ nodes: members, edgeCount });
}
clusterResults.sort((a, b) => b.edgeCount - a.edgeCount);
const topClusters = clusterResults.slice(0, 10);

// --- G. Layer List ---
const layersResult = {
  count: layers.length,
  list: layers.map(l => ({ id: l.id, name: l.name, description: l.description }))
};

// --- H. Node Summary Index ---
const nodeSummaryIndex = {};
for (const node of nodes) {
  nodeSummaryIndex[node.id] = { name: node.name, type: node.type, summary: node.summary || '' };
}

// --- Assemble output ---
const output = {
  scriptCompleted: true,
  entryPointCandidates,
  fanInRanking,
  fanOutRanking,
  bfsTraversal: bfsResult,
  nonCodeFiles,
  clusters: topClusters,
  layers: layersResult,
  nodeSummaryIndex,
  totalNodes: nodes.length,
  totalEdges: edges.length
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2), 'utf8');
  console.log(`Analysis complete. Wrote results to ${outputPath}`);
  process.exit(0);
} catch (err) {
  process.stderr.write(`Failed to write output: ${err.message}\n`);
  process.exit(1);
}
