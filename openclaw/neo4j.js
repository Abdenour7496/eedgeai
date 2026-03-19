#!/usr/bin/env node
/**
 * Neo4j driver CLI for OpenClaw.
 *
 * Uses the official neo4j-driver with a persistent connection pool and
 * exponential-backoff retry so every call is resilient to transient faults.
 *
 * Commands
 * --------
 *   neo4j-cli run    "<cypher>" ['<json-params>']
 *   neo4j-cli search "<text>"  [--limit N]
 *   neo4j-cli create "<label>" '<json-props>'
 *   neo4j-cli relate "<from-id>" "<type>" "<to-id>" ['<json-props>']
 *   neo4j-cli schema
 *   neo4j-cli stats
 *   neo4j-cli ping
 *
 * All output is JSON on stdout. Errors are JSON on stderr + exit 1.
 */

'use strict';

const neo4j = require('neo4j-driver');

const URI      = process.env.NEO4J_URI      || 'bolt://neo4j:7687';
const USER     = process.env.NEO4J_USER     || 'neo4j';
const PASSWORD = process.env.NEO4J_PASSWORD || 'test1234';
const DATABASE = process.env.NEO4J_DATABASE || 'neo4j';
const MAX_RETRY = parseInt(process.env.NEO4J_MAX_RETRIES || '3', 10);

// --- Driver (connection pool) ------------------------------------------------

const driver = neo4j.driver(
  URI,
  neo4j.auth.basic(USER, PASSWORD),
  {
    maxConnectionPoolSize:       10,
    connectionAcquisitionTimeout: 5_000,   // ms
    maxTransactionRetryTime:    15_000,    // ms – built-in retry for write txns
  }
);

// --- Retry helper ------------------------------------------------------------

async function withRetry(fn, retries = MAX_RETRY) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const retryable = err.code === 'ServiceUnavailable' ||
                        err.code === 'SessionExpired'    ||
                        err.code?.startsWith('Neo.TransientError');
      if (!retryable || attempt === retries - 1) throw err;
      await new Promise(r => setTimeout(r, Math.pow(2, attempt) * 500));
    }
  }
}

// --- Query helper ------------------------------------------------------------

async function runQuery(cypher, params = {}) {
  return withRetry(async () => {
    const session = driver.session({ database: DATABASE });
    try {
      const result = await session.run(cypher, params);
      return result.records.map(r => r.toObject());
    } finally {
      await session.close();
    }
  });
}

// --- Commands ----------------------------------------------------------------

const COMMANDS = {
  async ping() {
    await driver.verifyConnectivity();
    return { connected: true, uri: URI };
  },

  async run(args) {
    const [cypher, rawParams = '{}'] = args;
    if (!cypher) throw new Error('Usage: neo4j-cli run "<cypher>" [<json-params>]');
    return runQuery(cypher, JSON.parse(rawParams));
  },

  async search(args) {
    const limitIdx = args.indexOf('--limit');
    const limit    = limitIdx !== -1 ? parseInt(args[limitIdx + 1], 10) : 10;
    const text     = args.filter((_, i) => i !== limitIdx && i !== limitIdx + 1).join(' ');
    if (!text) throw new Error('Usage: neo4j-cli search "<text>" [--limit N]');

    return runQuery(
      'MATCH (n) ' +
      'WHERE any(k IN keys(n) WHERE toString(n[k]) CONTAINS $q) ' +
      'RETURN labels(n) AS labels, properties(n) AS props ' +
      'LIMIT $limit',
      { q: text.slice(0, 200), limit: neo4j.int(limit) }
    );
  },

  async create(args) {
    const [label, rawProps = '{}'] = args;
    if (!label) throw new Error('Usage: neo4j-cli create "<label>" \'<json-props>\'');
    const props = JSON.parse(rawProps);
    return runQuery(`CREATE (n:\`${label}\` $props) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props`, { props });
  },

  async relate(args) {
    const [fromId, type, toId, rawProps = '{}'] = args;
    if (!fromId || !type || !toId) throw new Error('Usage: neo4j-cli relate "<fromId>" "<TYPE>" "<toId>" [<json-props>]');
    const props = JSON.parse(rawProps);
    return runQuery(
      `MATCH (a) WHERE id(a) = $from MATCH (b) WHERE id(b) = $to ` +
      `CREATE (a)-[r:\`${type}\` $props]->(b) RETURN type(r) AS type, properties(r) AS props`,
      { from: neo4j.int(fromId), to: neo4j.int(toId), props }
    );
  },

  async schema() {
    const [labels, rels, props] = await Promise.all([
      runQuery('CALL db.labels() YIELD label RETURN collect(label) AS labels'),
      runQuery('CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types'),
      runQuery('CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS keys'),
    ]);
    return {
      labels:             labels[0]?.labels ?? [],
      relationship_types: rels[0]?.types   ?? [],
      property_keys:      props[0]?.keys   ?? [],
    };
  },

  async stats() {
    const [nodes, rels] = await Promise.all([
      runQuery('MATCH (n) RETURN count(n) AS count'),
      runQuery('MATCH ()-[r]->() RETURN count(r) AS count'),
    ]);
    return {
      node_count:         nodes[0]?.count?.toNumber?.() ?? nodes[0]?.count ?? 0,
      relationship_count: rels[0]?.count?.toNumber?.()  ?? rels[0]?.count  ?? 0,
    };
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
    if (cmd !== 'ping') await driver.verifyConnectivity();
    const result = await COMMANDS[cmd](rest);
    process.stdout.write(JSON.stringify(result, (_, v) =>
      typeof v?.toNumber === 'function' ? v.toNumber() : v
    , 2) + '\n');
  } catch (err) {
    process.stderr.write(JSON.stringify({ error: err.message, code: err.code }) + '\n');
    process.exit(1);
  } finally {
    await driver.close();
  }
})();
