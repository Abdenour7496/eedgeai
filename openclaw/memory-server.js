#!/usr/bin/env node
/**
 * memory-server.js — REST API + search UI for Claude Code memory entries.
 *
 * Routes
 * ------
 *   GET  /                          Search UI (HTML)
 *   GET  /api/search?q=&mode=&type= Semantic / graph / hybrid / compound search
 *   GET  /api/graph?type=&limit=    Browse Neo4j entries
 *   GET  /api/stats                 Counts + top-rated entries
 *   POST /api/feedback              Record relevance signal
 *   GET  /metrics                   Prometheus text exposition
 *
 * Auth
 *   Set MEMORY_API_KEY to require X-Api-Key header on /api/* routes.
 *   UI route is always open (no secrets in search results ≠ no key set).
 *
 * Env vars
 *   MEMORY_SERVER_PORT   default 4242
 *   MEMORY_API_KEY       optional — enables API key auth
 *   NEO4J_URI / QDRANT_URL / OPENAI_API_KEY / EMBEDDING_MODEL  (standard)
 */

'use strict';

const http    = require('http');
const https   = require('https');
const url     = require('url');
const crypto  = require('crypto');
const neo4j   = require('neo4j-driver');
const { QdrantClient } = require('@qdrant/js-client-rest');

// ── Config ────────────────────────────────────────────────────────────────────

const PORT         = parseInt(process.env.MEMORY_SERVER_PORT || '4242', 10);
const API_KEY      = process.env.MEMORY_API_KEY || '';
const NEO4J_URI    = process.env.NEO4J_URI       || 'bolt://localhost:7687';
const NEO4J_USER   = process.env.NEO4J_USER      || 'neo4j';
const NEO4J_PASS   = process.env.NEO4J_PASSWORD  || 'test1234';
const NEO4J_DB     = process.env.NEO4J_DATABASE  || 'neo4j';
const QDRANT_URL   = process.env.QDRANT_URL      || 'http://localhost:6333';
const OPENAI_KEY   = process.env.OPENAI_API_KEY  || '';
const EMBED_MODEL  = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
const COLLECTION   = 'claude-memory';

// ── Metrics (in-memory Prometheus counters) ───────────────────────────────────

const metrics = {
  requests_total:   0,
  search_total:     { semantic: 0, graph: 0, hybrid: 0, compound: 0 },
  feedback_total:   0,
  errors_total:     0,
  latency_sum_ms:   0,
  latency_count:    0,
};

function recordLatency(ms) {
  metrics.latency_sum_ms += ms;
  metrics.latency_count  += 1;
}

function prometheusText() {
  const avg = metrics.latency_count > 0
    ? (metrics.latency_sum_ms / metrics.latency_count).toFixed(1)
    : 0;
  return [
    '# HELP memory_requests_total Total HTTP requests to memory-server',
    '# TYPE memory_requests_total counter',
    `memory_requests_total ${metrics.requests_total}`,
    '',
    '# HELP memory_search_total Searches by mode',
    '# TYPE memory_search_total counter',
    ...Object.entries(metrics.search_total).map(
      ([mode, n]) => `memory_search_total{mode="${mode}"} ${n}`
    ),
    '',
    '# HELP memory_feedback_total Feedback signals recorded',
    '# TYPE memory_feedback_total counter',
    `memory_feedback_total ${metrics.feedback_total}`,
    '',
    '# HELP memory_errors_total API errors',
    '# TYPE memory_errors_total counter',
    `memory_errors_total ${metrics.errors_total}`,
    '',
    '# HELP memory_query_latency_avg_ms Average query latency',
    '# TYPE memory_query_latency_avg_ms gauge',
    `memory_query_latency_avg_ms ${avg}`,
    '',
  ].join('\n');
}

// ── Neo4j / Qdrant helpers ────────────────────────────────────────────────────

function makeDriver() {
  return neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USER, NEO4J_PASS));
}

async function runNeo4j(driver, cypher, params = {}) {
  const session = driver.session({ database: NEO4J_DB });
  try {
    const result = await session.run(cypher, params);
    return result.records.map(r => r.toObject());
  } finally {
    await session.close();
  }
}

function embedText(text) {
  return new Promise((resolve, reject) => {
    if (!OPENAI_KEY) return reject(new Error('OPENAI_API_KEY not set'));
    const body = JSON.stringify({ input: text.slice(0, 8000), model: EMBED_MODEL });
    const req = https.request({
      hostname: 'api.openai.com',
      path: '/v1/embeddings',
      method: 'POST',
      headers: {
        Authorization:   `Bearer ${OPENAI_KEY}`,
        'Content-Type':  'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    }, res => {
      let data = '';
      res.on('data', d => { data += d; });
      res.on('end', () => {
        try {
          const p = JSON.parse(data);
          if (p.error) return reject(new Error(p.error.message));
          resolve(p.data[0].embedding);
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ── Search handlers ───────────────────────────────────────────────────────────

async function searchSemantic(q, topK, typeFilter) {
  const qdrant = new QdrantClient({ url: QDRANT_URL });
  const vector = await embedText(q);
  const params = {
    vector, limit: topK, with_payload: true,
    ...(typeFilter && { filter: { must: [{ key: 'type', match: { value: typeFilter } }] } }),
  };
  const hits = await qdrant.search(COLLECTION, params);
  return hits.map(h => ({
    score: h.score, name: h.payload.name, type: h.payload.type,
    description: h.payload.description, body: h.payload.body,
    file_path: h.payload.file_path, neo4j_eid: h.payload.neo4j_element_id,
  }));
}

async function searchGraph(typeFilter, limit) {
  const driver = makeDriver();
  await driver.verifyConnectivity();
  try {
    const cypher = typeFilter
      ? `MATCH (m:MemoryEntry {type: $type})
         OPTIONAL MATCH (m)-[r]->(n:MemoryEntry)
         RETURN properties(m) AS entry,
                collect({rel: type(r), name: n.name, type: n.type}) AS relations
         LIMIT $limit`
      : `MATCH (m:MemoryEntry)
         OPTIONAL MATCH (m)-[r]->(n:MemoryEntry)
         RETURN properties(m) AS entry,
                collect({rel: type(r), name: n.name, type: n.type}) AS relations
         LIMIT $limit`;
    const rows = await runNeo4j(driver, cypher, { type: typeFilter, limit: neo4j.int(limit) });
    return rows.map(r => ({
      ...r.entry,
      relations: (r.relations || []).filter(x => x.name),
    }));
  } finally {
    await driver.close();
  }
}

async function searchHybrid(q, topK, typeFilter) {
  const hits = await searchSemantic(q, topK, typeFilter);
  if (!hits.length) return [];

  const driver = makeDriver();
  await driver.verifyConnectivity();
  try {
    return await Promise.all(hits.map(async hit => {
      if (!hit.neo4j_eid) return { ...hit, graph: { neighbours: [] } };
      const rows = await runNeo4j(driver,
        `MATCH (m:MemoryEntry) WHERE elementId(m) = $eid
         OPTIONAL MATCH (m)-[r]->(n:MemoryEntry)
         RETURN collect({rel: type(r), name: n.name, type: n.type}) AS neighbours`,
        { eid: hit.neo4j_eid }
      );
      return {
        ...hit,
        graph: { neighbours: (rows[0]?.neighbours || []).filter(nb => nb.name) },
      };
    }));
  } finally {
    await driver.close();
  }
}

async function searchCompound(q, topK, types, minScore) {
  const qdrant = new QdrantClient({ url: QDRANT_URL });
  const vector = await embedText(q);
  const params = {
    vector, limit: topK * 3, with_payload: true,
    ...(types.length > 0 && { filter: { must: [{ key: 'type', match: { any: types } }] } }),
  };
  const hits = (await qdrant.search(COLLECTION, params)).filter(h => h.score >= minScore);
  if (!hits.length) return [];

  const driver = makeDriver();
  await driver.verifyConnectivity();
  try {
    const eids = hits.map(h => h.payload.neo4j_element_id).filter(Boolean);
    const fbRows = eids.length > 0 ? await runNeo4j(driver,
      `UNWIND $eids AS eid
       MATCH (m:MemoryEntry) WHERE elementId(m) = eid
       OPTIONAL MATCH (f:QueryFeedback)-[:FOR]->(m)
       RETURN elementId(m) AS eid, avg(f.score) AS avg_fb`,
      { eids }
    ) : [];

    const fbMap = {};
    for (const r of fbRows) fbMap[r.eid] = r.avg_fb;

    return hits.slice(0, topK).map(h => {
      const fb      = fbMap[h.payload.neo4j_element_id] ?? null;
      const blended = fb !== null ? 0.6 * h.score + 0.4 * fb : h.score;
      return {
        score: parseFloat(blended.toFixed(4)),
        vector_score: h.score, feedback_score: fb,
        name: h.payload.name, type: h.payload.type,
        description: h.payload.description, body: h.payload.body,
        file_path: h.payload.file_path, neo4j_eid: h.payload.neo4j_element_id,
      };
    }).sort((a, b) => b.score - a.score);
  } finally {
    await driver.close();
  }
}

async function getStats() {
  const qdrant = new QdrantClient({ url: QDRANT_URL });
  const driver = makeDriver();
  await driver.verifyConnectivity();
  try {
    const [qdrantInfo, typeCounts, topRated] = await Promise.all([
      (async () => {
        try {
          const i = await qdrant.getCollection(COLLECTION);
          return { collection: COLLECTION, points: i.points_count, status: i.status };
        } catch { return { collection: COLLECTION, points: 0, status: 'missing' }; }
      })(),
      runNeo4j(driver,
        'MATCH (m:MemoryEntry) RETURN m.type AS type, count(m) AS count ORDER BY count DESC'
      ),
      runNeo4j(driver,
        `MATCH (f:QueryFeedback)-[:FOR]->(m:MemoryEntry)
         WITH m, avg(f.score) AS avg_score, count(f) AS votes WHERE votes >= 1
         RETURN m.name AS name, m.type AS type, avg_score, votes
         ORDER BY avg_score DESC LIMIT 5`
      ),
    ]);

    const toNum = v => typeof v?.toNumber === 'function' ? v.toNumber() : v;
    return {
      qdrant: qdrantInfo,
      neo4j:  { by_type: Object.fromEntries(typeCounts.map(r => [r.type, toNum(r.count)])) },
      top_rated: topRated.map(r => ({
        name: r.name, type: r.type,
        avg_score: toNum(r.avg_score), votes: toNum(r.votes),
      })),
      server_metrics: {
        requests_total:  metrics.requests_total,
        errors_total:    metrics.errors_total,
        avg_latency_ms:  metrics.latency_count > 0
          ? parseFloat((metrics.latency_sum_ms / metrics.latency_count).toFixed(1)) : 0,
      },
    };
  } finally {
    await driver.close();
  }
}

// ── HTML UI ───────────────────────────────────────────────────────────────────

const HTML = /* html */`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Memory Search</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;padding:24px}
  h1{font-size:1.4rem;font-weight:600;color:#7dd3fc;margin-bottom:20px}
  .bar{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap}
  input[type=text]{flex:1;min-width:220px;background:#1e2535;border:1px solid #334155;
    border-radius:6px;padding:9px 12px;color:#e2e8f0;font-size:.95rem;outline:none}
  input[type=text]:focus{border-color:#7dd3fc}
  select{background:#1e2535;border:1px solid #334155;border-radius:6px;
    padding:9px 10px;color:#e2e8f0;font-size:.9rem;outline:none}
  button{background:#2563eb;color:#fff;border:none;border-radius:6px;
    padding:9px 18px;cursor:pointer;font-size:.9rem;font-weight:500}
  button:hover{background:#1d4ed8}
  .chip{display:inline-block;font-size:.72rem;font-weight:600;padding:2px 8px;
    border-radius:99px;margin-right:4px;text-transform:uppercase;letter-spacing:.04em}
  .chip-project{background:#1e3a5f;color:#7dd3fc}
  .chip-user{background:#1a3a2a;color:#6ee7b7}
  .chip-feedback{background:#3a2a1a;color:#fbbf24}
  .chip-reference{background:#2a1a3a;color:#c084fc}
  .card{background:#1e2535;border:1px solid #334155;border-radius:8px;
    padding:16px 18px;margin-bottom:12px}
  .card-header{display:flex;align-items:center;gap:8px;margin-bottom:8px}
  .card-name{font-weight:600;font-size:1rem}
  .card-score{margin-left:auto;font-size:.8rem;color:#94a3b8}
  .card-desc{color:#94a3b8;font-size:.85rem;margin-bottom:8px}
  .card-body{font-size:.83rem;color:#cbd5e1;white-space:pre-wrap;
    max-height:200px;overflow-y:auto;line-height:1.5}
  .relations{margin-top:10px;display:flex;gap:6px;flex-wrap:wrap}
  .rel-tag{background:#1a2540;border:1px solid #334155;border-radius:4px;
    font-size:.72rem;padding:2px 7px;color:#94a3b8}
  .fb-row{display:flex;gap:8px;margin-top:10px;align-items:center}
  .fb-row span{font-size:.75rem;color:#64748b}
  .fb-btn{background:none;border:1px solid #334155;border-radius:4px;
    padding:3px 10px;cursor:pointer;font-size:.8rem;color:#94a3b8}
  .fb-btn:hover{border-color:#7dd3fc;color:#7dd3fc}
  .status{font-size:.82rem;color:#64748b;margin-bottom:12px}
  .error{color:#f87171;font-size:.85rem;padding:12px;
    background:#2a1a1a;border-radius:6px;margin-bottom:12px}
  #results{min-height:80px}
</style>
</head>
<body>
<h1>Memory Search</h1>

<div class="bar">
  <input id="q" type="text" placeholder="Search memories…" autofocus>
  <select id="mode">
    <option value="hybrid">Hybrid</option>
    <option value="semantic">Semantic</option>
    <option value="compound">Compound</option>
    <option value="graph">Graph browse</option>
  </select>
  <select id="type">
    <option value="">All types</option>
    <option value="project">Project</option>
    <option value="user">User</option>
    <option value="feedback">Feedback</option>
    <option value="reference">Reference</option>
  </select>
  <button onclick="search()">Search</button>
</div>

<div id="status" class="status"></div>
<div id="results"></div>

<script>
const API_KEY = window.__API_KEY__ || '';

async function apiFetch(path, opts = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (API_KEY) headers['X-Api-Key'] = API_KEY;
  const r = await fetch(path, { ...opts, headers: { ...headers, ...opts.headers } });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function chip(type) {
  return \`<span class="chip chip-\${type}">\${type}</span>\`;
}

function scoreBar(score) {
  const pct = Math.round((score || 0) * 100);
  return \`<span class="card-score">\${pct}%</span>\`;
}

function renderResult(item, idx) {
  const rels = (item.relations || item.graph?.neighbours || [])
    .filter(r => r.name)
    .map(r => \`<span class="rel-tag">\${r.rel || r.type} → \${r.name}</span>\`)
    .join('');

  const fbId = item.neo4j_eid || item.neo4j_element_id || '';

  return \`
  <div class="card">
    <div class="card-header">
      \${chip(item.type)}
      <span class="card-name">\${item.name}</span>
      \${scoreBar(item.score)}
    </div>
    \${item.description ? \`<div class="card-desc">\${item.description}</div>\` : ''}
    <div class="card-body">\${item.body || ''}</div>
    \${rels ? \`<div class="relations">\${rels}</div>\` : ''}
    \${fbId ? \`
    <div class="fb-row">
      <span>Relevant?</span>
      <button class="fb-btn" onclick="sendFeedback('\${fbId}', 1, event)">👍 Yes</button>
      <button class="fb-btn" onclick="sendFeedback('\${fbId}', 0, event)">👎 No</button>
    </div>\` : ''}
  </div>\`;
}

async function search() {
  const q    = document.getElementById('q').value.trim();
  const mode = document.getElementById('mode').value;
  const type = document.getElementById('type').value;
  const out  = document.getElementById('results');
  const stat = document.getElementById('status');

  if (!q && mode !== 'graph') { stat.textContent = 'Enter a query.'; return; }

  stat.textContent = 'Searching…';
  out.innerHTML = '';

  const t0 = Date.now();
  try {
    const params = new URLSearchParams({ q, mode });
    if (type) params.set('type', type);
    const data = await apiFetch('/api/search?' + params);
    const ms = Date.now() - t0;
    stat.textContent = \`\${data.length} result(s) in \${ms}ms\`;
    out.innerHTML = data.length
      ? data.map(renderResult).join('')
      : '<div class="status">No results.</div>';
  } catch (e) {
    stat.textContent = '';
    out.innerHTML = \`<div class="error">\${e.message}</div>\`;
  }
}

async function sendFeedback(eid, score, ev) {
  const btn = ev.target;
  btn.disabled = true;
  const q = document.getElementById('q').value.trim();
  try {
    await apiFetch('/api/feedback', {
      method: 'POST',
      body: JSON.stringify({ neo4j_eid: eid, score, query: q }),
    });
    btn.textContent = score === 1 ? '✓ Thanks!' : '✓ Noted';
  } catch { btn.disabled = false; }
}

// Search on Enter
document.getElementById('q').addEventListener('keydown', e => {
  if (e.key === 'Enter') search();
});
</script>
</body>
</html>`;

// ── HTTP server ───────────────────────────────────────────────────────────────

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => { data += chunk; });
    req.on('end', () => {
      try { resolve(data ? JSON.parse(data) : {}); }
      catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}

function send(res, status, body, contentType = 'application/json') {
  const payload = contentType === 'application/json'
    ? JSON.stringify(body, (_, v) => typeof v?.toNumber === 'function' ? v.toNumber() : v, 2)
    : body;
  res.writeHead(status, { 'Content-Type': contentType, 'Access-Control-Allow-Origin': '*' });
  res.end(payload);
}

function authOk(req) {
  if (!API_KEY) return true;
  return req.headers['x-api-key'] === API_KEY ||
    new url.URL(req.url, 'http://x').searchParams.get('key') === API_KEY;
}

const server = http.createServer(async (req, res) => {
  metrics.requests_total++;
  const t0      = Date.now();
  const parsed  = new url.URL(req.url, `http://localhost:${PORT}`);
  const pathname = parsed.pathname;

  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, { 'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type,X-Api-Key',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS' });
    return res.end();
  }

  try {
    // ── UI
    if (pathname === '/' && req.method === 'GET') {
      // Inject API key as JS variable so the UI can auth without exposing it in URL
      const html = HTML.replace('window.__API_KEY__ || \'\'',
        API_KEY ? `'${API_KEY}'` : "''");
      return send(res, 200, html, 'text/html');
    }

    // ── Metrics (no auth — Prometheus scraper shouldn't need it)
    if (pathname === '/metrics' && req.method === 'GET') {
      recordLatency(Date.now() - t0);
      return send(res, 200, prometheusText(), 'text/plain; version=0.0.4');
    }

    // ── API auth gate
    if (pathname.startsWith('/api/') && !authOk(req)) {
      return send(res, 401, { error: 'Unauthorized — set X-Api-Key header' });
    }

    // ── GET /api/search
    if (pathname === '/api/search' && req.method === 'GET') {
      const q        = parsed.searchParams.get('q') || '';
      const mode     = parsed.searchParams.get('mode') || 'hybrid';
      const typeFilter = parsed.searchParams.get('type') || '';
      const topK     = parseInt(parsed.searchParams.get('top_k') || '5', 10);
      const minScore = parseFloat(parsed.searchParams.get('min_score') || '0');
      const typesRaw = parsed.searchParams.get('types') || '';
      const types    = typesRaw ? typesRaw.split(',').map(t => t.trim()) : [];

      if (metrics.search_total[mode] !== undefined) metrics.search_total[mode]++;

      let results;
      switch (mode) {
        case 'semantic':  results = await searchSemantic(q, topK, typeFilter); break;
        case 'graph':     results = await searchGraph(typeFilter || null, topK * 4); break;
        case 'hybrid':    results = await searchHybrid(q, topK, typeFilter); break;
        case 'compound':  results = await searchCompound(q, topK, types.length ? types : (typeFilter ? [typeFilter] : []), minScore); break;
        default: return send(res, 400, { error: `Unknown mode: ${mode}` });
      }

      recordLatency(Date.now() - t0);
      return send(res, 200, results);
    }

    // ── GET /api/graph
    if (pathname === '/api/graph' && req.method === 'GET') {
      const typeFilter = parsed.searchParams.get('type') || null;
      const limit     = parseInt(parsed.searchParams.get('limit') || '20', 10);
      const results   = await searchGraph(typeFilter, limit);
      recordLatency(Date.now() - t0);
      return send(res, 200, results);
    }

    // ── GET /api/stats
    if (pathname === '/api/stats' && req.method === 'GET') {
      const stats = await getStats();
      recordLatency(Date.now() - t0);
      return send(res, 200, stats);
    }

    // ── POST /api/feedback
    if (pathname === '/api/feedback' && req.method === 'POST') {
      const body = await parseBody(req);
      const { neo4j_eid, score, query: q = '' } = body;

      if (!neo4j_eid || score === undefined) {
        return send(res, 400, { error: 'neo4j_eid and score are required' });
      }
      const s = parseFloat(score);
      if (isNaN(s) || s < 0 || s > 1) {
        return send(res, 400, { error: 'score must be 0–1' });
      }

      const driver = makeDriver();
      await driver.verifyConnectivity();
      try {
        const now = new Date().toISOString();
        const rows = await runNeo4j(driver,
          `MATCH (m:MemoryEntry) WHERE elementId(m) = $eid
           CREATE (f:QueryFeedback {score: $score, query: $q, created_at: $now})-[:FOR]->(m)
           RETURN elementId(f) AS feid, m.name AS entry_name`,
          { eid: neo4j_eid, score: s, q, now }
        );
        if (!rows.length) return send(res, 404, { error: 'MemoryEntry not found' });
        metrics.feedback_total++;
        recordLatency(Date.now() - t0);
        return send(res, 200, {
          recorded: true,
          feedback_id: rows[0].feid,
          entry_name: rows[0].entry_name,
          score: s,
        });
      } finally {
        await driver.close();
      }
    }

    send(res, 404, { error: 'Not found' });

  } catch (err) {
    metrics.errors_total++;
    console.error('[memory-server]', err.message);
    recordLatency(Date.now() - t0);
    send(res, 500, { error: err.message });
  }
});

server.listen(PORT, () => {
  console.log(`[memory-server] Listening on http://localhost:${PORT}`);
  console.log(`[memory-server] Neo4j : ${NEO4J_URI}`);
  console.log(`[memory-server] Qdrant: ${QDRANT_URL}`);
  if (API_KEY) console.log('[memory-server] API key auth enabled');
});
