#!/usr/bin/env node
/**
 * memory-sync.js — Sync Claude Code MEMORY.md files into Neo4j + Qdrant.
 *
 * Reads the MEMORY.md index, parses each linked .md file (frontmatter + body),
 * creates/updates MemoryEntry nodes in Neo4j, and upserts embeddings into the
 * `claude-memory` Qdrant collection.  Safe to re-run — uses file path as
 * idempotent key for both stores.
 *
 * Usage
 * -----
 *   node memory-sync.js [--memory-dir <path>] [--dry-run]
 *
 * Env vars (inherit from .env / docker-compose convention)
 *   NEO4J_URI         bolt://localhost:7687
 *   NEO4J_USER        neo4j
 *   NEO4J_PASSWORD    test1234
 *   QDRANT_URL        http://localhost:6333
 *   OPENAI_API_KEY    required for embeddings
 *   EMBEDDING_MODEL   text-embedding-3-small
 */

'use strict';

const fs    = require('fs');
const path  = require('path');
const https = require('https');
const crypto = require('crypto');
const neo4j  = require('neo4j-driver');
const { QdrantClient } = require('@qdrant/js-client-rest');

// ── Config ────────────────────────────────────────────────────────────────────

const NEO4J_URI      = process.env.NEO4J_URI       || 'bolt://localhost:7687';
const NEO4J_USER     = process.env.NEO4J_USER      || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD  || 'test1234';
const NEO4J_DATABASE = process.env.NEO4J_DATABASE  || 'neo4j';
const QDRANT_URL     = process.env.QDRANT_URL      || 'http://localhost:6333';
const OPENAI_KEY     = process.env.OPENAI_API_KEY  || '';
const EMBED_MODEL    = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
const COLLECTION     = 'claude-memory';

const DEFAULT_MEMORY_DIR = path.join(
  process.env.USERPROFILE || process.env.HOME || '',
  '.claude', 'projects',
  'c--Users-abden-Documents-eedgeai',
  'memory'
);

// ── CLI args ──────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const memDirIdx = argv.indexOf('--memory-dir');
const MEMORY_DIR = memDirIdx !== -1 ? argv[memDirIdx + 1] : DEFAULT_MEMORY_DIR;
const DRY_RUN    = argv.includes('--dry-run');

// ── Frontmatter parser ────────────────────────────────────────────────────────

function parseFrontmatter(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (!match) return { meta: {}, body: content.trim() };

  const meta = {};
  for (const line of match[1].split(/\r?\n/)) {
    const colon = line.indexOf(':');
    if (colon === -1) continue;
    const key = line.slice(0, colon).trim();
    const val = line.slice(colon + 1).trim();
    meta[key] = val;
  }
  return { meta, body: match[2].trim() };
}

// ── Parse MEMORY.md for linked files ─────────────────────────────────────────

function parseMemoryIndex(memoryDir) {
  const indexPath = path.join(memoryDir, 'MEMORY.md');
  if (!fs.existsSync(indexPath)) {
    throw new Error(`MEMORY.md not found at: ${indexPath}`);
  }

  const content = fs.readFileSync(indexPath, 'utf8');
  const files = [];

  // Match markdown links: [text](filename.md)
  for (const match of content.matchAll(/\[([^\]]+)\]\(([^)]+\.md)\)/g)) {
    const [, linkText, filename] = match;
    const filePath = path.join(memoryDir, filename);
    if (fs.existsSync(filePath) && filename !== 'MEMORY.md') {
      files.push({ linkText, filename, filePath });
    }
  }

  return files;
}

// ── Load and parse a memory file ──────────────────────────────────────────────

function loadMemoryFile(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const { meta, body } = parseFrontmatter(raw);
  return {
    name:        meta.name        || path.basename(filePath, '.md'),
    description: meta.description || '',
    type:        meta.type        || 'project',
    body,
    filePath,
    fileHash:    crypto.createHash('sha256').update(raw).digest('hex').slice(0, 16),
  };
}

// ── Embedding ─────────────────────────────────────────────────────────────────

function embedText(text) {
  return new Promise((resolve, reject) => {
    if (!OPENAI_KEY) return reject(new Error('OPENAI_API_KEY is not set'));
    const body = JSON.stringify({ input: text.slice(0, 8000), model: EMBED_MODEL });
    const req = https.request(
      {
        hostname: 'api.openai.com',
        path: '/v1/embeddings',
        method: 'POST',
        headers: {
          Authorization:  `Bearer ${OPENAI_KEY}`,
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
      },
      res => {
        let data = '';
        res.on('data', d => { data += d; });
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            if (parsed.error) return reject(new Error(parsed.error.message));
            resolve(parsed.data[0].embedding);
          } catch (e) { reject(e); }
        });
      }
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ── Neo4j ─────────────────────────────────────────────────────────────────────

async function neo4jUpsertMemory(session, entry) {
  const now = new Date().toISOString();

  // Upsert MemoryEntry node keyed on filePath
  const result = await session.run(
    `MERGE (m:MemoryEntry {file_path: $filePath})
     ON CREATE SET
       m.name        = $name,
       m.description = $description,
       m.type        = $type,
       m.body        = $body,
       m.file_hash   = $fileHash,
       m.created_at  = $now,
       m.updated_at  = $now
     ON MATCH SET
       m.name        = $name,
       m.description = $description,
       m.type        = $type,
       m.body        = $body,
       m.file_hash   = $fileHash,
       m.updated_at  = $now
     RETURN elementId(m) AS eid, m.file_hash AS hash`,
    {
      filePath:    entry.filePath,
      name:        entry.name,
      description: entry.description,
      type:        entry.type,
      body:        entry.body,
      fileHash:    entry.fileHash,
      now,
    }
  );

  const record = result.records[0];
  return { eid: record.get('eid'), hash: record.get('hash') };
}

async function neo4jCreateRelationships(session, entries) {
  // Link all MemoryEntry nodes to a single MemoryIndex node
  await session.run(
    `MERGE (idx:MemoryIndex {name: 'claude-memory'})
     WITH idx
     MATCH (m:MemoryEntry)
     MERGE (idx)-[:CONTAINS]->(m)`
  );

  // Link entries of the same type with SAME_TYPE edges
  await session.run(
    `MATCH (a:MemoryEntry), (b:MemoryEntry)
     WHERE a.type = b.type AND a.file_path <> b.file_path
     MERGE (a)-[:SAME_TYPE]->(b)`
  );

  // Link project memories to reference memories (project USES reference)
  await session.run(
    `MATCH (p:MemoryEntry {type: 'project'}), (r:MemoryEntry {type: 'reference'})
     MERGE (p)-[:REFERENCES]->(r)`
  );

  // Link feedback memories to user memories (feedback APPLIES_TO user context)
  await session.run(
    `MATCH (f:MemoryEntry {type: 'feedback'}), (u:MemoryEntry {type: 'user'})
     MERGE (f)-[:APPLIES_TO]->(u)`
  );

  // RELATED_TO: keyword overlap — entries sharing ≥2 significant tokens
  // Computed in JS, written as Cypher MERGE statements
  const keywords = entries.map(e => ({
    filePath: e.filePath,
    tokens: tokenize(`${e.name} ${e.description} ${e.body}`),
  }));

  for (let i = 0; i < keywords.length; i++) {
    for (let j = i + 1; j < keywords.length; j++) {
      const shared = keywords[i].tokens.filter(t => keywords[j].tokens.has(t));
      if (shared.length >= 2) {
        await session.run(
          `MATCH (a:MemoryEntry {file_path: $a}), (b:MemoryEntry {file_path: $b})
           MERGE (a)-[:RELATED_TO {shared_keywords: $kw}]->(b)`,
          { a: keywords[i].filePath, b: keywords[j].filePath, kw: shared.slice(0, 10) }
        );
      }
    }
  }

  // SUPERSEDES: body contains explicit supersession signal ("supersedes", "replaces", "updated")
  // and mentions another entry's name
  for (const entry of entries) {
    const bodyLower = entry.body.toLowerCase();
    const supersedes = /supersedes?|replaces?|updated?\s+from/.test(bodyLower);
    if (!supersedes) continue;

    for (const other of entries) {
      if (other.filePath === entry.filePath) continue;
      if (bodyLower.includes(other.name.toLowerCase())) {
        await session.run(
          `MATCH (a:MemoryEntry {file_path: $a}), (b:MemoryEntry {file_path: $b})
           MERGE (a)-[:SUPERSEDES]->(b)`,
          { a: entry.filePath, b: other.filePath }
        );
      }
    }
  }
}

// ── Keyword tokenizer ─────────────────────────────────────────────────────────

const STOP_WORDS = new Set([
  'the','a','an','and','or','of','to','in','is','it','that','this',
  'for','with','as','at','by','from','be','are','was','were','have',
  'has','had','not','on','do','if','so','but','use','used','using',
  'will','can','all','any','one','new','also','may','then','than',
  'its','our','we','you','i','me','my','your','how','what','which',
]);

function tokenize(text) {
  return new Set(
    text.toLowerCase()
      .replace(/[^a-z0-9\s_-]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 3 && !STOP_WORDS.has(t))
  );
}

// ── Qdrant ────────────────────────────────────────────────────────────────────

async function qdrantEnsureCollection(client, vectorSize) {
  try {
    await client.getCollection(COLLECTION);
  } catch {
    await client.createCollection(COLLECTION, {
      vectors: { size: vectorSize, distance: 'Cosine' },
    });
    log(`Created Qdrant collection: ${COLLECTION}`);
  }
}

function filePathToUuid(filePath) {
  const h = crypto.createHash('md5').update(filePath).digest('hex');
  return `${h.slice(0,8)}-${h.slice(8,12)}-${h.slice(12,16)}-${h.slice(16,20)}-${h.slice(20,32)}`;
}

async function qdrantUpsertMemory(client, entry, neo4jEid) {
  const text  = `${entry.name}\n${entry.description}\n\n${entry.body}`;
  const vector = await embedText(text);

  await qdrantEnsureCollection(client, vector.length);

  await client.upsert(COLLECTION, {
    wait: true,
    points: [{
      id:     filePathToUuid(entry.filePath),
      vector,
      payload: {
        name:            entry.name,
        description:     entry.description,
        type:            entry.type,
        body:            entry.body,
        file_path:       entry.filePath,
        file_hash:       entry.fileHash,
        neo4j_element_id: neo4jEid,
        node_type:       'MemoryEntry',
        synced_at:       new Date().toISOString(),
      },
    }],
  });

  return vector.length;
}

// ── Logging ───────────────────────────────────────────────────────────────────

function log(msg) { process.stderr.write(`[memory-sync] ${msg}\n`); }

// ── Main ──────────────────────────────────────────────────────────────────────

(async () => {
  try {
    log(`Memory dir : ${MEMORY_DIR}`);
    log(`Neo4j      : ${NEO4J_URI}`);
    log(`Qdrant     : ${QDRANT_URL}`);
    if (DRY_RUN) log('DRY RUN — no writes');

    // Parse index
    const files = parseMemoryIndex(MEMORY_DIR);
    if (files.length === 0) {
      log('No memory files found in MEMORY.md index');
      process.exit(0);
    }
    log(`Found ${files.length} memory file(s): ${files.map(f => f.filename).join(', ')}`);

    // Load all entries
    const entries = files.map(f => loadMemoryFile(f.filePath));

    if (DRY_RUN) {
      process.stdout.write(JSON.stringify(entries.map(e => ({
        name: e.name, type: e.type, description: e.description, bodyLength: e.body.length,
      })), null, 2) + '\n');
      process.exit(0);
    }

    // Neo4j
    const driver  = neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USER, NEO4J_PASSWORD));
    await driver.verifyConnectivity();
    log('Neo4j connected');

    const session = driver.session({ database: NEO4J_DATABASE });
    const eids = {};

    try {
      for (const entry of entries) {
        const { eid } = await neo4jUpsertMemory(session, entry);
        eids[entry.filePath] = eid;
        log(`Neo4j upserted: ${entry.name} (${entry.type})`);
      }
      await neo4jCreateRelationships(session, entries);
      log('Neo4j relationships created');
    } finally {
      await session.close();
      await driver.close();
    }

    // Qdrant
    const qdrant = new QdrantClient({ url: QDRANT_URL });
    for (const entry of entries) {
      const dims = await qdrantUpsertMemory(qdrant, entry, eids[entry.filePath]);
      log(`Qdrant upserted: ${entry.name} (${dims}d vector)`);
    }

    const result = {
      status:  'ok',
      synced:  entries.length,
      collection: COLLECTION,
      entries: entries.map(e => ({ name: e.name, type: e.type, neo4j_eid: eids[e.filePath] })),
    };
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');

  } catch (err) {
    process.stderr.write(JSON.stringify({ error: err.message }) + '\n');
    process.exit(1);
  }
})();
