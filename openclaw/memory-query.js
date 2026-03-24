#!/usr/bin/env node
/**
 * memory-query.js — Hybrid semantic + graph retrieval for Claude Code memories.
 *
 * Commands
 * --------
 *   node memory-query.js semantic "<query>" [--top-k N]
 *       Vector search in Qdrant `claude-memory` collection.
 *
 *   node memory-query.js graph [--type user|feedback|project|reference] [--limit N]
 *       Retrieve MemoryEntry nodes from Neo4j, optionally filtered by type.
 *
 *   node memory-query.js hybrid "<query>" [--top-k N] [--type <type>]
 *       Qdrant semantic search → fetch linked Neo4j nodes for graph context.
 *       Returns merged result: vector score + graph neighbours.
 *
 *   node memory-query.js compound "<query>" [--types t1,t2] [--min-score 0.7] [--top-k N]
 *       Multi-type filtered semantic search with score threshold.
 *       Feedback scores from Neo4j are mixed in to boost proven entries.
 *
 *   node memory-query.js feedback <neo4j_eid> <score 0-1> ["<query>"]
 *       Record a relevance signal for a memory entry.
 *       Stored as QueryFeedback nodes; factored in by `compound`.
 *
 *   node memory-query.js stats
 *       Show counts in both stores plus top-rated entries by feedback.
 *
 * All output is JSON on stdout. Errors are JSON on stderr + exit 1.
 *
 * Env vars: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, QDRANT_URL, OPENAI_API_KEY, EMBEDDING_MODEL
 */

'use strict';

const https  = require('https');
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

// ── Neo4j helpers ─────────────────────────────────────────────────────────────

function makeDriver() {
  return neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USER, NEO4J_PASSWORD));
}

async function runQuery(driver, cypher, params = {}) {
  const session = driver.session({ database: NEO4J_DATABASE });
  try {
    const result = await session.run(cypher, params);
    return result.records.map(r => r.toObject());
  } finally {
    await session.close();
  }
}

// ── Commands ──────────────────────────────────────────────────────────────────

const COMMANDS = {

  async semantic(args) {
    const topKIdx = args.indexOf('--top-k');
    const topK    = topKIdx !== -1 ? parseInt(args[topKIdx + 1], 10) : 5;
    const rest    = args.filter((_, i) => i !== topKIdx && i !== topKIdx + 1);
    const query   = rest.join(' ');
    if (!query) throw new Error('Usage: memory-query.js semantic "<query>" [--top-k N]');

    const qdrant = new QdrantClient({ url: QDRANT_URL });
    const vector = await embedText(query);

    const hits = await qdrant.search(COLLECTION, {
      vector,
      limit: topK,
      with_payload: true,
    });

    return hits.map(h => ({
      score:       h.score,
      name:        h.payload.name,
      type:        h.payload.type,
      description: h.payload.description,
      body:        h.payload.body,
      file_path:   h.payload.file_path,
      neo4j_eid:   h.payload.neo4j_element_id,
    }));
  },

  async graph(args) {
    const typeIdx  = args.indexOf('--type');
    const type     = typeIdx !== -1 ? args[typeIdx + 1] : null;
    const limitIdx = args.indexOf('--limit');
    const limit    = limitIdx !== -1 ? parseInt(args[limitIdx + 1], 10) : 20;

    const driver = makeDriver();
    await driver.verifyConnectivity();
    try {
      const cypher = type
        ? `MATCH (m:MemoryEntry {type: $type})
           OPTIONAL MATCH (m)-[r]->(n:MemoryEntry)
           RETURN properties(m) AS entry,
                  collect({type: type(r), name: n.name, ntype: n.type}) AS relations
           LIMIT $limit`
        : `MATCH (m:MemoryEntry)
           OPTIONAL MATCH (m)-[r]->(n:MemoryEntry)
           RETURN properties(m) AS entry,
                  collect({type: type(r), name: n.name, ntype: n.type}) AS relations
           LIMIT $limit`;

      const rows = await runQuery(driver, cypher,
        { type, limit: neo4j.int(limit) }
      );

      return rows.map(row => {
        const e = row.entry;
        return {
          name:        e.name,
          type:        e.type,
          description: e.description,
          file_path:   e.file_path,
          updated_at:  e.updated_at,
          relations:   (row.relations || []).filter(r => r.name),
        };
      });
    } finally {
      await driver.close();
    }
  },

  async hybrid(args) {
    const topKIdx  = args.indexOf('--top-k');
    const topK     = topKIdx !== -1 ? parseInt(args[topKIdx + 1], 10) : 5;
    const typeIdx  = args.indexOf('--type');
    const typeFilter = typeIdx !== -1 ? args[typeIdx + 1] : null;
    const rest     = args.filter((_, i) =>
      i !== topKIdx && i !== topKIdx + 1 &&
      i !== typeIdx  && i !== typeIdx + 1
    );
    const query = rest.join(' ');
    if (!query) throw new Error('Usage: memory-query.js hybrid "<query>" [--top-k N] [--type <type>]');

    // 1. Vector search
    const qdrant = new QdrantClient({ url: QDRANT_URL });
    const vector = await embedText(query);

    let searchParams = { vector, limit: topK, with_payload: true };
    if (typeFilter) {
      searchParams.filter = { must: [{ key: 'type', match: { value: typeFilter } }] };
    }
    const hits = await qdrant.search(COLLECTION, searchParams);

    if (hits.length === 0) return [];

    // 2. Graph context: for each hit, fetch the node + its 1-hop neighbours
    const driver = makeDriver();
    await driver.verifyConnectivity();
    try {
      const results = [];
      for (const hit of hits) {
        const eid = hit.payload.neo4j_element_id;
        let graphContext = { entry: null, neighbours: [] };

        if (eid) {
          const rows = await runQuery(driver,
            `MATCH (m:MemoryEntry) WHERE elementId(m) = $eid
             OPTIONAL MATCH (m)-[r]->(n:MemoryEntry)
             RETURN properties(m) AS entry,
                    collect({rel: type(r), name: n.name, type: n.type}) AS neighbours`,
            { eid }
          );
          if (rows.length > 0) {
            graphContext.entry      = rows[0].entry;
            graphContext.neighbours = (rows[0].neighbours || []).filter(nb => nb.name);
          }
        }

        results.push({
          score:        hit.score,
          name:         hit.payload.name,
          type:         hit.payload.type,
          description:  hit.payload.description,
          body:         hit.payload.body,
          file_path:    hit.payload.file_path,
          graph:        graphContext,
        });
      }
      return results;
    } finally {
      await driver.close();
    }
  },

  /**
   * compound "<query>" [--types t1,t2,...] [--min-score 0.7] [--top-k N]
   *
   * Multi-type filtered semantic search.  Results below --min-score are
   * dropped.  Each result's final score is blended with its average
   * QueryFeedback score from Neo4j (60/40 vector/feedback split).
   */
  async compound(args) {
    const topKIdx    = args.indexOf('--top-k');
    const topK       = topKIdx !== -1 ? parseInt(args[topKIdx + 1], 10) : 10;
    const typesIdx   = args.indexOf('--types');
    const typeList   = typesIdx !== -1 ? args[typesIdx + 1].split(',').map(t => t.trim()) : [];
    const scoreIdx   = args.indexOf('--min-score');
    const minScore   = scoreIdx !== -1 ? parseFloat(args[scoreIdx + 1]) : 0.0;
    const rest       = args.filter((_, i) =>
      i !== topKIdx  && i !== topKIdx  + 1 &&
      i !== typesIdx && i !== typesIdx + 1 &&
      i !== scoreIdx && i !== scoreIdx + 1
    );
    const query = rest.join(' ');
    if (!query) throw new Error(
      'Usage: memory-query.js compound "<query>" [--types t1,t2] [--min-score 0.7] [--top-k N]'
    );

    // 1. Vector search with optional type filter
    const qdrant = new QdrantClient({ url: QDRANT_URL });
    const vector = await embedText(query);

    const searchParams = {
      vector,
      limit: topK * 3,   // fetch extra, we'll prune by min-score
      with_payload: true,
      ...(typeList.length > 0 && {
        filter: {
          must: [{ key: 'type', match: { any: typeList } }],
        },
      }),
    };

    const hits = await qdrant.search(COLLECTION, searchParams);
    const filtered = hits.filter(h => h.score >= minScore);
    if (filtered.length === 0) return [];

    // 2. Pull Neo4j feedback scores for these entries
    const driver = makeDriver();
    await driver.verifyConnectivity();
    try {
      const eids = filtered.map(h => h.payload.neo4j_element_id).filter(Boolean);
      let feedbackMap = {};

      if (eids.length > 0) {
        const fbRows = await runQuery(driver,
          `UNWIND $eids AS eid
           MATCH (m:MemoryEntry) WHERE elementId(m) = eid
           OPTIONAL MATCH (f:QueryFeedback)-[:FOR]->(m)
           RETURN elementId(m) AS eid,
                  avg(f.score) AS avg_feedback,
                  count(f)     AS feedback_count`,
          { eids }
        );
        for (const row of fbRows) {
          feedbackMap[row.eid] = {
            avg: row.avg_feedback ?? null,
            count: typeof row.feedback_count?.toNumber === 'function'
              ? row.feedback_count.toNumber() : (row.feedback_count ?? 0),
          };
        }
      }

      // 3. Blend scores: if feedback exists, 60% vector + 40% feedback
      const results = filtered.slice(0, topK).map(hit => {
        const eid       = hit.payload.neo4j_element_id;
        const fb        = feedbackMap[eid] || { avg: null, count: 0 };
        const blended   = fb.avg !== null
          ? 0.6 * hit.score + 0.4 * fb.avg
          : hit.score;

        return {
          score:          parseFloat(blended.toFixed(4)),
          vector_score:   hit.score,
          feedback_score: fb.avg,
          feedback_count: fb.count,
          name:           hit.payload.name,
          type:           hit.payload.type,
          description:    hit.payload.description,
          body:           hit.payload.body,
          file_path:      hit.payload.file_path,
          neo4j_eid:      eid,
        };
      });

      // Re-sort by blended score descending
      return results.sort((a, b) => b.score - a.score);
    } finally {
      await driver.close();
    }
  },

  /**
   * feedback <neo4j_eid> <score 0-1> ["<query context>"]
   *
   * Records a QueryFeedback node in Neo4j linked to the MemoryEntry.
   * score=1 means highly relevant, score=0 means irrelevant.
   */
  async feedback(args) {
    const [eid, rawScore, ...queryParts] = args;
    if (!eid || rawScore === undefined) {
      throw new Error('Usage: memory-query.js feedback <neo4j_eid> <score 0-1> ["<query>"]');
    }
    const score = parseFloat(rawScore);
    if (isNaN(score) || score < 0 || score > 1) {
      throw new Error('score must be a number between 0 and 1');
    }
    const queryCtx = queryParts.join(' ') || '';

    const driver = makeDriver();
    await driver.verifyConnectivity();
    try {
      const now = new Date().toISOString();
      const rows = await runQuery(driver,
        `MATCH (m:MemoryEntry) WHERE elementId(m) = $eid
         CREATE (f:QueryFeedback {
           score:      $score,
           query:      $query,
           created_at: $now
         })-[:FOR]->(m)
         RETURN elementId(f) AS feid, m.name AS entry_name`,
        { eid, score, query: queryCtx, now }
      );

      if (rows.length === 0) throw new Error(`No MemoryEntry found with elementId: ${eid}`);
      return {
        recorded:   true,
        feedback_id: rows[0].feid,
        entry_name:  rows[0].entry_name,
        score,
      };
    } finally {
      await driver.close();
    }
  },

  async stats() {
    const qdrant = new QdrantClient({ url: QDRANT_URL });
    const driver = makeDriver();
    await driver.verifyConnectivity();

    try {
      const [qdrantInfo, neo4jCounts] = await Promise.all([
        (async () => {
          try {
            const info = await qdrant.getCollection(COLLECTION);
            return { collection: COLLECTION, points: info.points_count, status: info.status };
          } catch {
            return { collection: COLLECTION, points: 0, status: 'missing' };
          }
        })(),
        runQuery(driver,
          `MATCH (m:MemoryEntry)
           RETURN m.type AS type, count(m) AS count
           ORDER BY count DESC`
        ),
      ]);

      const typeCounts = {};
      for (const row of neo4jCounts) {
        typeCounts[row.type] = typeof row.count?.toNumber === 'function'
          ? row.count.toNumber() : row.count;
      }

      // Top-rated entries by average feedback score
      const topRated = await runQuery(driver,
        `MATCH (f:QueryFeedback)-[:FOR]->(m:MemoryEntry)
         WITH m, avg(f.score) AS avg_score, count(f) AS votes
         WHERE votes >= 1
         RETURN m.name AS name, m.type AS type, avg_score, votes
         ORDER BY avg_score DESC
         LIMIT 5`
      );

      return {
        qdrant:    qdrantInfo,
        neo4j:     { by_type: typeCounts },
        top_rated: topRated.map(r => ({
          name:      r.name,
          type:      r.type,
          avg_score: typeof r.avg_score?.toNumber === 'function' ? r.avg_score.toNumber() : r.avg_score,
          votes:     typeof r.votes?.toNumber    === 'function' ? r.votes.toNumber()    : r.votes,
        })),
      };
    } finally {
      await driver.close();
    }
  },
};

// ── Main ──────────────────────────────────────────────────────────────────────

(async () => {
  const [,, cmd, ...rest] = process.argv;

  if (!cmd || !COMMANDS[cmd]) {
    const available = Object.keys(COMMANDS).join(', ');
    process.stderr.write(JSON.stringify({ error: `Unknown command. Available: ${available}` }) + '\n');
    process.exit(1);
  }

  try {
    const result = await COMMANDS[cmd](rest);
    process.stdout.write(JSON.stringify(result, (_, v) =>
      typeof v?.toNumber === 'function' ? v.toNumber() : v
    , 2) + '\n');
  } catch (err) {
    process.stderr.write(JSON.stringify({ error: err.message }) + '\n');
    process.exit(1);
  }
})();
