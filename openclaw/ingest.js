#!/usr/bin/env node
/**
 * Document ingest CLI for OpenClaw — GCOR pipeline.
 *
 * Reads a file (or stdin), chunks the text, creates Document + Chunk nodes
 * in Neo4j, embeds each chunk, and upserts it to Qdrant with the Neo4j
 * elementId so the GCOR proxy can traverse from vector hits to graph context.
 *
 * Supported formats
 * -----------------
 *   .txt  .md  .json  .csv   — plain text / UTF-8
 *   .pdf                     — pdf-parse (text layer only)
 *   .docx                    — mammoth
 *
 * Usage
 * -----
 *   ingest-cli <file>                       ingest a file
 *   ingest-cli <file> --title "My Doc"      override document title
 *   ingest-cli <file> --agent-id "a1"       scope to an agent partition
 *   ingest-cli <file> --access-level restricted
 *   ingest-cli <file> --collection myIndex  target a different Qdrant collection
 *   ingest-cli <file> --chunk-size 1500     chars per chunk (default 2000)
 *   ingest-cli <file> --chunk-overlap 200   overlap chars (default 200)
 *   echo "text" | ingest-cli --stdin --title "pasted text"
 *
 * All output is JSON on stdout. Errors are JSON on stderr + exit 1.
 */

'use strict';

const fs      = require('fs');
const path    = require('path');
const https   = require('https');
const crypto  = require('crypto');
const neo4j   = require('neo4j-driver');
const { QdrantClient } = require('@qdrant/js-client-rest');

// ── Config ───────────────────────────────────────────────────────────────────

const NEO4J_URI       = process.env.NEO4J_URI        || 'bolt://neo4j:7687';
const NEO4J_USER      = process.env.NEO4J_USER        || 'neo4j';
const NEO4J_PASSWORD  = process.env.NEO4J_PASSWORD    || 'test1234';
const NEO4J_DATABASE  = process.env.NEO4J_DATABASE    || 'neo4j';
const QDRANT_URL      = process.env.QDRANT_URL        || 'http://qdrant:6333';
const OPENAI_API_KEY  = process.env.OPENAI_API_KEY    || '';
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL   || 'text-embedding-3-small';
const DEFAULT_ACCESS  = process.env.DEFAULT_ACCESS_LEVEL || 'public';
const DEFAULT_COLLECTION = process.env.QDRANT_COLLECTION || 'documents';

// ── CLI args ─────────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const args = { flags: {}, positional: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      args.flags[key] = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true;
    } else {
      args.positional.push(a);
    }
  }
  return args;
}

const { flags, positional } = parseArgs(process.argv.slice(2));
const filePath      = positional[0] || null;
const useStdin      = !!flags.stdin;
const title         = flags.title         || (filePath ? path.basename(filePath) : 'Untitled');
const agentId       = flags.agentId       || '';
const accessLevel   = flags.accessLevel   || DEFAULT_ACCESS;
const collection    = flags.collection    || DEFAULT_COLLECTION;
const CHUNK_SIZE    = parseInt(flags.chunkSize    || '2000', 10);
const CHUNK_OVERLAP = parseInt(flags.chunkOverlap || '200',  10);

// Image-specific flags
if (flags.visionBackend)  process.env.VISION_BACKEND  = flags.visionBackend;
if (flags.visionModel)    process.env.VISION_MODEL    = flags.visionModel;
if (flags.noVision)       process.env.NO_VISION       = '1';

// ── Text extraction ───────────────────────────────────────────────────────────

const { extractImageText, SUPPORTED_EXTENSIONS: IMAGE_EXTS, isDicom } =
  require('./image-extractor');

// Holds structured metadata set during image extraction; read by main()
let _imageMetadata = null;

async function extractText(fp, rawBuffer) {
  const ext = (fp ? path.extname(fp) : '.txt').toLowerCase();

  // ── Images (regular, DICOM, NIfTI) ──────────────────────────────────────────
  // Dispatch on extension OR on DICOM magic bytes (files with no / wrong ext)
  const isImageExt  = IMAGE_EXTS.includes(ext) || ext === '.nii';
  const isDicomFile = isDicom(rawBuffer);

  if (isImageExt || isDicomFile) {
    const result = await extractImageText(rawBuffer, ext, fp || '');
    _imageMetadata = result.metadata;
    return result.text;
  }

  if (ext === '.pdf') {
    try {
      const pdfParse = require('pdf-parse');
      const data = await pdfParse(rawBuffer);
      return data.text;
    } catch (e) {
      throw new Error(`PDF parse failed: ${e.message}. Ensure pdf-parse is installed.`);
    }
  }

  if (ext === '.docx') {
    try {
      const mammoth = require('mammoth');
      const result = await mammoth.extractRawText({ buffer: rawBuffer });
      return result.value;
    } catch (e) {
      throw new Error(`DOCX parse failed: ${e.message}. Ensure mammoth is installed.`);
    }
  }

  if (ext === '.json') {
    try {
      const obj = JSON.parse(rawBuffer.toString('utf8'));
      return JSON.stringify(obj, null, 2);
    } catch {
      return rawBuffer.toString('utf8');
    }
  }

  // .txt .md .csv and everything else — treat as UTF-8 text
  return rawBuffer.toString('utf8');
}

// ── Chunking ──────────────────────────────────────────────────────────────────

function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = start + size;
    // Try to break at a sentence or word boundary
    if (end < text.length) {
      const nl = text.lastIndexOf('\n', end);
      const sp = text.lastIndexOf(' ', end);
      const boundary = nl > start + size * 0.5 ? nl : sp > start + size * 0.5 ? sp : end;
      end = boundary;
    }
    const chunk = text.slice(start, end).trim();
    if (chunk.length > 0) chunks.push(chunk);
    start = end - overlap;
    if (start <= 0 && chunks.length > 0) break;
  }
  return chunks;
}

// ── Embedding ─────────────────────────────────────────────────────────────────

function embedTexts(texts) {
  return new Promise((resolve, reject) => {
    if (!OPENAI_API_KEY) return reject(new Error('OPENAI_API_KEY is not set'));
    const body = JSON.stringify({ input: texts, model: EMBEDDING_MODEL });
    const req = https.request(
      {
        hostname: 'api.openai.com',
        path: '/v1/embeddings',
        method: 'POST',
        headers: {
          Authorization: `Bearer ${OPENAI_API_KEY}`,
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
            // Return embeddings in index order
            const sorted = parsed.data.sort((a, b) => a.index - b.index);
            resolve(sorted.map(d => d.embedding));
          } catch (e) { reject(e); }
        });
      }
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ── Neo4j ────────────────────────────────────────────────────────────────────

async function neo4jIngest(documentId, title, source, chunks, agentId, accessLevel, imageProps = {}) {
  const driver = neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USER, NEO4J_PASSWORD));
  try {
    await driver.verifyConnectivity();
    const session = driver.session({ database: NEO4J_DATABASE });
    try {
      const now = new Date().toISOString();

      // Upsert the Document node, then merge any image-specific metadata fields
      const docResult = await session.run(
        `MERGE (d:Document {document_id: $document_id})
         ON CREATE SET d.created_at = $created_at
         SET d.title        = $title,
             d.source       = $source,
             d.agent_id     = $agent_id,
             d.access_level = $access_level,
             d.chunk_count  = $chunk_count
         SET d += $image_props
         RETURN elementId(d) AS eid`,
        {
          document_id:  documentId,
          title,
          source,
          agent_id:     agentId,
          access_level: accessLevel,
          chunk_count:  neo4j.int(chunks.length),
          created_at:   now,
          image_props:  imageProps,
        }
      );
      const docEid = docResult.records[0].get('eid');

      // Upsert Chunk nodes and link them with HAS_CHUNK
      const chunkEids = [];
      for (let i = 0; i < chunks.length; i++) {
        const chunkId = `${documentId}-chunk-${i}`;
        const chunkResult = await session.run(
          `MATCH (d:Document {document_id: $document_id})
           MERGE (c:Chunk {chunk_id: $chunk_id})
           ON CREATE SET c.created_at = $created_at
           SET c.text           = $text,
               c.position       = $position,
               c.document_id    = $document_id,
               c.document_title = $title,
               c.agent_id       = $agent_id,
               c.access_level   = $access_level,
               c.confidence     = 1.0
           MERGE (d)-[:HAS_CHUNK]->(c)
           RETURN elementId(c) AS eid`,
          {
            document_id: documentId,
            chunk_id:    chunkId,
            text:        chunks[i],
            position:    neo4j.int(i),
            title,
            agent_id:    agentId,
            access_level: accessLevel,
            created_at:  now,
          }
        );
        chunkEids.push(chunkResult.records[0].get('eid'));
      }

      // Link consecutive chunks with :NEXT for traversal
      for (let i = 0; i < chunkEids.length - 1; i++) {
        await session.run(
          `MATCH (a:Chunk) WHERE elementId(a) = $a
           MATCH (b:Chunk) WHERE elementId(b) = $b
           MERGE (a)-[:NEXT]->(b)`,
          { a: chunkEids[i], b: chunkEids[i + 1] }
        );
      }

      return { docEid, chunkEids };
    } finally {
      await session.close();
    }
  } finally {
    await driver.close();
  }
}

// ── Qdrant ───────────────────────────────────────────────────────────────────

async function qdrantIngest(collection, chunks, chunkEids, documentId, title, agentId, accessLevel) {
  const client = new QdrantClient({ url: QDRANT_URL });
  const now    = new Date().toISOString();

  // Batch embed (max 96 texts per OpenAI call)
  const BATCH = 96;
  const allVectors = [];
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const vecs  = await embedTexts(batch);
    allVectors.push(...vecs);
  }

  // Ensure collection exists
  try {
    await client.getCollection(collection);
  } catch {
    await client.createCollection(collection, {
      vectors: { size: allVectors[0].length, distance: 'Cosine' },
    });
  }

  // Upsert points
  const points = chunks.map((text, i) => ({
    id:      md5Uuid(chunkEids[i]),
    vector:  allVectors[i],
    payload: {
      text,
      neo4j_element_id: chunkEids[i],
      node_type:        'Chunk',
      document_id:      documentId,
      document_title:   title,
      position:         i,
      agent_id:         agentId,
      access_level:     accessLevel,
      confidence:       1.0,
      valid_from:       now,
      valid_to:         null,
    },
  }));

  await client.upsert(collection, { points, wait: true });
  return points.length;
}

// ── UUID from MD5 (Qdrant needs UUID format) ──────────────────────────────────

function md5Uuid(str) {
  const h = crypto.createHash('md5').update(str).digest('hex');
  return `${h.slice(0,8)}-${h.slice(8,12)}-${h.slice(12,16)}-${h.slice(16,20)}-${h.slice(20,32)}`;
}

// ── Main ─────────────────────────────────────────────────────────────────────

(async () => {
  try {
    if (!filePath && !useStdin) {
      const help = [
        'Usage: ingest-cli <file> [options]',
        '       echo "text" | ingest-cli --stdin --title "My Doc"',
        '',
        'Supported formats:',
        '  Text/docs  .txt .md .csv .json .pdf .docx',
        '  Images     .jpg .jpeg .png .gif .bmp .webp .tiff .avif',
        '  Medical    .dcm .dicom (DICOM)   .nii .nii.gz (NIfTI)',
        '             DICOM files without extension auto-detected by magic bytes',
        '',
        'Options:',
        '  --title <str>            Document title (default: filename)',
        '  --agent-id <str>         Agent partition (default: shared)',
        '  --access-level <str>     public | restricted | agent:<id> (default: public)',
        '  --collection <str>       Qdrant collection (default: documents)',
        '  --chunk-size <n>         Characters per chunk (default: 2000)',
        '  --chunk-overlap <n>      Overlap characters (default: 200)',
        '  --stdin                  Read from stdin instead of a file',
        '  --vision-backend <str>   openai (default) | anthropic',
        '  --vision-model <str>     Override vision model (default: gpt-4o)',
        '  --no-vision              Skip vision API call; store metadata only',
      ];
      process.stderr.write(JSON.stringify({ error: 'No input specified', usage: help }) + '\n');
      process.exit(1);
    }

    // Read input
    let rawBuffer;
    let source;
    if (useStdin) {
      const chunks = [];
      for await (const chunk of process.stdin) chunks.push(chunk);
      rawBuffer = Buffer.concat(chunks);
      source    = 'stdin';
    } else {
      rawBuffer = fs.readFileSync(filePath);
      source    = path.resolve(filePath);
    }

    // Extract text
    const text = (await extractText(filePath, rawBuffer)).trim();
    if (!text) {
      process.stderr.write(JSON.stringify({ error: 'No text extracted from input' }) + '\n');
      process.exit(1);
    }

    // Chunk
    const textChunks = chunkText(text);
    if (textChunks.length === 0) {
      process.stderr.write(JSON.stringify({ error: 'Text produced no chunks' }) + '\n');
      process.exit(1);
    }

    const documentId = md5Uuid(`${source}-${Date.now()}`).replace(/-/g, '').slice(0, 16);

    process.stderr.write(`[ingest] "${title}" → ${textChunks.length} chunks\n`);
    if (_imageMetadata) {
      process.stderr.write(`[ingest] Image type: ${_imageMetadata.format}` +
        (_imageMetadata.modality ? ` / ${_imageMetadata.modality}` : '') + '\n');
    }

    // Flatten image metadata into Neo4j-safe scalar props (no nested objects)
    const imageProps = {};
    if (_imageMetadata) {
      for (const [k, v] of Object.entries(_imageMetadata)) {
        if (v === null || v === undefined) continue;
        imageProps[`image_${k}`] = Array.isArray(v) ? v.join(',') : String(v);
      }
    }

    // Neo4j
    process.stderr.write('[ingest] Writing to Neo4j...\n');
    const { docEid, chunkEids } = await neo4jIngest(
      documentId, title, source, textChunks, agentId, accessLevel, imageProps
    );

    // Qdrant
    process.stderr.write('[ingest] Embedding and writing to Qdrant...\n');
    const upserted = await qdrantIngest(collection, textChunks, chunkEids, documentId, title, agentId, accessLevel);

    const result = {
      status:        'ok',
      document_id:   documentId,
      document_eid:  docEid,
      title,
      source,
      chunks:        textChunks.length,
      qdrant_points: upserted,
      collection,
      ..._imageMetadata && { image_metadata: _imageMetadata },
    };
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  } catch (err) {
    process.stderr.write(JSON.stringify({ error: err.message }) + '\n');
    process.exit(1);
  }
})();
