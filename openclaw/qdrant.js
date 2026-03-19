#!/usr/bin/env node
/**
 * Qdrant client CLI for OpenClaw.
 *
 * Uses the official @qdrant/js-client-rest with exponential-backoff retry.
 *
 * Commands
 * --------
 *   qdrant-cli list
 *   qdrant-cli info    <collection>
 *   qdrant-cli count   <collection>
 *   qdrant-cli search  <collection> "<query-text>" [--top-k N]   (needs OPENAI_API_KEY)
 *   qdrant-cli upsert  <collection> "<text>" ['<json-metadata>']  (needs OPENAI_API_KEY)
 *   qdrant-cli delete  <collection>
 *   qdrant-cli ping
 *
 * All output is JSON on stdout. Errors are JSON on stderr + exit 1.
 */

'use strict';

const { QdrantClient } = require('@qdrant/js-client-rest');
const https = require('https');

const QDRANT_URL      = process.env.QDRANT_URL      || 'http://quarant:6333';
const OPENAI_API_KEY  = process.env.OPENAI_API_KEY  || '';
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
const MAX_RETRY       = parseInt(process.env.QDRANT_MAX_RETRIES || '3', 10);

const client = new QdrantClient({ url: QDRANT_URL });

// --- Retry helper ------------------------------------------------------------

async function withRetry(fn, retries = MAX_RETRY) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === retries - 1) throw err;
      await new Promise(r => setTimeout(r, Math.pow(2, attempt) * 500));
    }
  }
}

// --- Embedding ---------------------------------------------------------------

function embed(text) {
  return new Promise((resolve, reject) => {
    if (!OPENAI_API_KEY) {
      return reject(new Error('OPENAI_API_KEY is not set'));
    }
    const body = JSON.stringify({ input: text.slice(0, 8000), model: EMBEDDING_MODEL });
    const req = https.request(
      {
        hostname: 'api.openai.com',
        path: '/v1/embeddings',
        method: 'POST',
        headers: {
          Authorization:  `Bearer ${OPENAI_API_KEY}`,
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

// --- Ensure collection exists ------------------------------------------------

async function ensureCollection(name, vectorSize) {
  try {
    await client.getCollection(name);
  } catch {
    await client.createCollection(name, {
      vectors: { size: vectorSize, distance: 'Cosine' },
    });
  }
}

// --- Commands ----------------------------------------------------------------

const COMMANDS = {
  async ping() {
    const result = await withRetry(() => client.getCollections());
    return { connected: true, url: QDRANT_URL, collections: result.collections.length };
  },

  async list() {
    const result = await withRetry(() => client.getCollections());
    return result.collections.map(c => c.name);
  },

  async info(args) {
    const [collection] = args;
    if (!collection) throw new Error('Usage: qdrant-cli info <collection>');
    const info = await withRetry(() => client.getCollection(collection));
    return {
      name:          collection,
      vectors_count: info.vectors_count,
      points_count:  info.points_count,
      status:        info.status,
      config:        info.config,
    };
  },

  async count(args) {
    const [collection] = args;
    if (!collection) throw new Error('Usage: qdrant-cli count <collection>');
    const result = await withRetry(() => client.count(collection));
    return { collection, count: result.count };
  },

  async search(args) {
    const topKIdx  = args.indexOf('--top-k');
    const topK     = topKIdx !== -1 ? parseInt(args[topKIdx + 1], 10) : 5;
    const rest     = args.filter((_, i) => i !== topKIdx && i !== topKIdx + 1);
    const [collection, ...queryParts] = rest;
    const query    = queryParts.join(' ');

    if (!collection || !query) throw new Error('Usage: qdrant-cli search <collection> "<query>" [--top-k N]');

    const vector  = await embed(query);
    const results = await withRetry(() => client.search(collection, {
      vector,
      limit: topK,
      with_payload: true,
    }));

    return results.map(r => ({ score: r.score, payload: r.payload }));
  },

  async upsert(args) {
    const [collection, text, rawMeta = '{}'] = args;
    if (!collection || !text) throw new Error('Usage: qdrant-cli upsert <collection> "<text>" [<json-metadata>]');

    const vector = await embed(text);
    await ensureCollection(collection, vector.length);

    const id = crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
    const result = await withRetry(() => client.upsert(collection, {
      points: [{ id, vector, payload: { text, ...JSON.parse(rawMeta) } }],
    }));

    return { status: result.status, id };
  },

  async delete(args) {
    const [collection] = args;
    if (!collection) throw new Error('Usage: qdrant-cli delete <collection>');
    const result = await withRetry(() => client.deleteCollection(collection));
    return { deleted: result };
  },
};

// --- Main --------------------------------------------------------------------

(async () => {
  const [,, cmd, ...rest] = process.argv;

  if (!cmd || !COMMANDS[cmd]) {
    const available = Object.keys(COMMANDS).join(', ');
    process.stderr.write(JSON.stringify({ error: `Unknown command. Available: ${available}` }) + '\n');
    process.exit(1);
  }

  try {
    const result = await COMMANDS[cmd](rest);
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  } catch (err) {
    process.stderr.write(JSON.stringify({ error: err.message }) + '\n');
    process.exit(1);
  }
})();
