#!/usr/bin/env node
/**
 * memory-watch.js — File-change daemon for the Claude Code memory system.
 *
 * Polls the memory directory every POLL_INTERVAL_MS milliseconds.
 * When any .md file is added, modified, or removed it triggers a full
 * memory-sync and emits a structured change log to stdout.
 *
 * Usage
 * -----
 *   node memory-watch.js [--memory-dir <path>] [--interval <ms>]
 *
 * Env vars
 *   MEMORY_POLL_INTERVAL_MS   default 5000
 *   NEO4J_URI / QDRANT_URL    passed through to memory-sync
 *
 * Output
 *   Each sync event is JSON on stdout:
 *     { event: "change"|"add"|"remove", file, triggeredAt, syncResult }
 *   Heartbeats every 60 s:
 *     { event: "heartbeat", watchedFiles, lastSyncAt }
 */

'use strict';

const fs     = require('fs');
const path   = require('path');
const crypto = require('crypto');
const { execFile } = require('child_process');

// ── Config ────────────────────────────────────────────────────────────────────

const argv        = process.argv.slice(2);
const dirIdx      = argv.indexOf('--memory-dir');
const intIdx      = argv.indexOf('--interval');

const DEFAULT_MEMORY_DIR = path.join(
  process.env.USERPROFILE || process.env.HOME || '',
  '.claude', 'projects',
  'c--Users-abden-Documents-eedgeai',
  'memory'
);

const MEMORY_DIR  = dirIdx  !== -1 ? argv[dirIdx  + 1] : DEFAULT_MEMORY_DIR;
const POLL_MS     = intIdx  !== -1 ? parseInt(argv[intIdx + 1], 10)
                                   : parseInt(process.env.MEMORY_POLL_INTERVAL_MS || '5000', 10);
const SYNC_SCRIPT = path.join(__dirname, 'memory-sync.js');

// ── State ─────────────────────────────────────────────────────────────────────

/** Map<filename, sha256> — tracks last known hash of each .md file */
const knownHashes = new Map();
let lastSyncAt    = null;
let syncInFlight  = false;

// ── Helpers ───────────────────────────────────────────────────────────────────

function fileHash(filePath) {
  try {
    return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
  } catch {
    return null;
  }
}

function out(obj) {
  process.stdout.write(JSON.stringify({ ...obj, ts: new Date().toISOString() }) + '\n');
}

function err(msg) {
  process.stderr.write(`[memory-watch] ${msg}\n`);
}

// ── Sync trigger ──────────────────────────────────────────────────────────────

function runSync(reason) {
  if (syncInFlight) {
    err(`Sync already running, skipping (reason: ${reason})`);
    return;
  }
  syncInFlight = true;
  err(`Triggering sync — ${reason}`);

  const env = { ...process.env };  // inherits NEO4J_*, QDRANT_URL, OPENAI_API_KEY
  const args = ['--memory-dir', MEMORY_DIR];

  execFile(process.execPath, [SYNC_SCRIPT, ...args], { env }, (error, stdout, stderr) => {
    syncInFlight = false;
    lastSyncAt   = new Date().toISOString();

    if (error) {
      err(`Sync failed: ${error.message}`);
      out({ event: 'sync_error', reason, error: error.message });
      return;
    }

    let syncResult = {};
    try { syncResult = JSON.parse(stdout); } catch { syncResult = { raw: stdout.trim() }; }

    out({ event: 'synced', reason, syncResult });
    err(`Sync complete — ${syncResult.synced ?? '?'} entries`);
  });
}

// ── Poll ──────────────────────────────────────────────────────────────────────

function scanDir() {
  let mdFiles;
  try {
    mdFiles = fs.readdirSync(MEMORY_DIR).filter(f => f.endsWith('.md'));
  } catch (e) {
    err(`Cannot read memory dir: ${e.message}`);
    return;
  }

  const changes = [];

  // Detect additions and modifications
  for (const file of mdFiles) {
    const fullPath = path.join(MEMORY_DIR, file);
    const hash     = fileHash(fullPath);
    const prev     = knownHashes.get(file);

    if (!prev) {
      changes.push({ event: 'add', file });
    } else if (prev !== hash) {
      changes.push({ event: 'change', file });
    }

    knownHashes.set(file, hash);
  }

  // Detect removals
  for (const [file] of knownHashes) {
    if (!mdFiles.includes(file)) {
      changes.push({ event: 'remove', file });
      knownHashes.delete(file);
    }
  }

  if (changes.length > 0) {
    for (const c of changes) out({ ...c, triggeredAt: new Date().toISOString() });
    runSync(changes.map(c => `${c.event}:${c.file}`).join(', '));
  }
}

// ── Heartbeat ─────────────────────────────────────────────────────────────────

function heartbeat() {
  out({
    event:        'heartbeat',
    watchedFiles: knownHashes.size,
    memoryDir:    MEMORY_DIR,
    pollMs:       POLL_MS,
    lastSyncAt,
  });
}

// ── Startup ───────────────────────────────────────────────────────────────────

err(`Watching ${MEMORY_DIR} every ${POLL_MS}ms`);

// Seed initial hashes (no sync on first scan — files are assumed already synced)
try {
  for (const file of fs.readdirSync(MEMORY_DIR).filter(f => f.endsWith('.md'))) {
    knownHashes.set(file, fileHash(path.join(MEMORY_DIR, file)));
  }
  err(`Tracking ${knownHashes.size} file(s): ${[...knownHashes.keys()].join(', ')}`);
} catch (e) {
  err(`Startup error: ${e.message}`);
  process.exit(1);
}

// Emit initial heartbeat
heartbeat();

// Poll loop
setInterval(scanDir, POLL_MS);

// Heartbeat every 60 s
setInterval(heartbeat, 60_000);

// Graceful shutdown
for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => {
    err(`Received ${sig}, shutting down`);
    process.exit(0);
  });
}
