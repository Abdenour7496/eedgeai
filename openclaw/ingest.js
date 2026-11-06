#!/usr/bin/env node
/**
 * Document ingest CLI for OpenClaw — GCOR pipeline.
 *
 * Reads a file (or stdin), chunks the text, and queues each chunk as a Graphiti
 * episode. Graphiti extracts temporal entities and facts into FalkorDB.
 * Also persists the original input to MinIO (best-effort — see
 * storeOriginalToS3) under the same originals/<document_id>/<filename>
 * convention proxy/main.py uses, so a document is findable in object storage
 * regardless of which ingestion path it came through.
 *
 * Supported formats
 * -----------------
 *   .txt  .md  .json  .csv   — plain text / UTF-8
 *   .pdf                     — unpdf (text layer only)
 *   .docx                    — mammoth
 *
 * Usage
 * -----
 *   ingest-cli <file>                       ingest a file
 *   ingest-cli <file> --title "My Doc"      override document title
 *   ingest-cli <file> --agent-id "a1"       scope to an agent partition
 *   ingest-cli <file> --access-level restricted
 *   ingest-cli <file> --collection myGroup  target a different Graphiti group
 *   ingest-cli <file> --chunk-size 1500     chars per chunk (default 2000)
 *   ingest-cli <file> --chunk-overlap 200   overlap chars (default 200)
 *   echo "text" | ingest-cli --stdin --title "pasted text"
 *
 * All output is JSON on stdout. Errors are JSON on stderr + exit 1.
 */

'use strict';

const fs      = require('fs');
const path    = require('path');
const http    = require('http');
const https   = require('https');
const crypto  = require('crypto');
const { Client: MinioClient } = require('minio');

// ── Config ───────────────────────────────────────────────────────────────────

const DEFAULT_ACCESS  = process.env.DEFAULT_ACCESS_LEVEL || 'public';
const DEFAULT_COLLECTION = process.env.GRAPHITI_GROUP_ID || 'documents';
const GRAPHITI_URL        = (process.env.GRAPHITI_URL || 'http://graphiti:8000').replace(/\/$/, '');

// Document storage (MinIO, S3-compatible) — same bucket/key convention as
// proxy/main.py's _store_original_to_s3, so a document ingested via either
// path lands in the same place. Best-effort: ingestion still succeeds if
// MinIO is unreachable or unconfigured, it just skips persisting the original.
const S3_ENDPOINT_URL     = process.env.S3_ENDPOINT_URL     || 'http://seaweedfs:8333';
const S3_ACCESS_KEY       = process.env.S3_ACCESS_KEY       || '';
const S3_SECRET_KEY       = process.env.S3_SECRET_KEY       || '';
const S3_BUCKET           = process.env.S3_BUCKET           || 'documents';
const S3_ORIGINALS_PREFIX = process.env.S3_ORIGINALS_PREFIX || 'originals/';

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
      // unpdf, not pdf-parse: pdf-parse@1.1.4 vendors a hardcoded 2018-era
      // pdfjs-dist (v1.10.100) that breaks on ANY PDF — not just malformed
      // ones — once Node's own 'http' module has been loaded anywhere in the
      // process (true here: MinIO and the image extractor load it before this
      // code runs), throwing
      // "bad XRef entry" on perfectly valid input. unpdf wraps a current,
      // actively-maintained pdfjs-dist build with no such issue.
      const { extractText: extractPdfText, getDocumentProxy } = require('unpdf');
      const pdf = await getDocumentProxy(new Uint8Array(rawBuffer));
      const { text } = await extractPdfText(pdf, { mergePages: true });
      return text;
    } catch (e) {
      throw new Error(`PDF parse failed: ${e.message}. Ensure unpdf is installed.`);
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

// ── MinIO (S3-compatible object storage) ───────────────────────────────────────

let _minioClient = null;
let _minioClientInit = false;

function getMinioClient() {
  if (!_minioClientInit) {
    _minioClientInit = true;
    if (S3_ACCESS_KEY && S3_SECRET_KEY) {
      const u = new URL(S3_ENDPOINT_URL);
      _minioClient = new MinioClient({
        endPoint: u.hostname,
        port:     u.port ? parseInt(u.port, 10) : (u.protocol === 'https:' ? 443 : 80),
        useSSL:   u.protocol === 'https:',
        accessKey: S3_ACCESS_KEY,
        secretKey: S3_SECRET_KEY,
      });
    }
  }
  return _minioClient;
}

// Persist the raw input bytes to MinIO, creating the bucket if needed.
// Returns the object key, or null on failure/unconfigured — mirrors
// proxy/main.py's _store_original_to_s3 (same key scheme, same
// best-effort semantics) so documents ingested via either path are
// findable under the same originals/<document_id>/<filename> convention.
async function storeOriginalToS3(documentId, filename, buffer, bucket = S3_BUCKET) {
  const client = getMinioClient();
  if (!client) return null;
  const key = `${S3_ORIGINALS_PREFIX}${documentId}/${filename}`;
  try {
    const exists = await client.bucketExists(bucket).catch(() => false);
    if (!exists) {
      try {
        await client.makeBucket(bucket);
      } catch (err) {
        if (!/already own|BucketAlready/i.test(err.message || '')) throw err;
      }
    }
    await client.putObject(bucket, key, buffer);
    return key;
  } catch (err) {
    process.stderr.write(
      `[ingest] S3 store of original file failed (doc_id=${documentId}, bucket=${bucket}): ${err.message}\n`
    );
    return null;
  }
}

// ── Stable UUID ───────────────────────────────────────────────────────────────

function md5Uuid(str) {
  const h = crypto.createHash('md5').update(str).digest('hex');
  return `${h.slice(0,8)}-${h.slice(8,12)}-${h.slice(12,16)}-${h.slice(16,20)}-${h.slice(20,32)}`;
}

async function graphitiIngest(groupId, chunks, documentId, title, source, agentId, accessLevel) {
  const timestamp = new Date().toISOString();
  const messages = chunks.map((content, position) => ({
    name: `${title} — chunk ${position + 1}`,
    role_type: 'system',
    role: 'document',
    timestamp,
    source_description: source,
    content: `Document: ${title}\nSource: ${source}\nDocument ID: ${documentId}\n` +
      `Chunk: ${position + 1}/${chunks.length}\nAgent: ${agentId || 'shared'}\n` +
      `Access: ${accessLevel}\n\n${content}`,
  }));
  const response = await fetch(`${GRAPHITI_URL}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group_id: groupId, messages }),
  });
  if (!response.ok) throw new Error(`Graphiti ingest failed (${response.status}): ${await response.text()}`);
  return messages.length;
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
        '  --collection <str>       Graphiti group (default: documents)',
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

    // Flatten image metadata for storage metadata (no nested objects).
    const imageProps = {};
    if (_imageMetadata) {
      for (const [k, v] of Object.entries(_imageMetadata)) {
        if (v === null || v === undefined) continue;
        imageProps[`image_${k}`] = Array.isArray(v) ? v.join(',') : String(v);
      }
    }

    // MinIO — best-effort original-file copy, same convention proxy/main.py uses
    const storageFilename = filePath
      ? path.basename(filePath)
      : `${title.replace(/[^\w.\-]+/g, '_') || 'stdin'}.txt`;
    const storageKey = await storeOriginalToS3(documentId, storageFilename, rawBuffer);
    if (storageKey) {
      imageProps.storage_bucket = S3_BUCKET;
      imageProps.storage_key    = storageKey;
    }

    process.stderr.write('[ingest] Queuing episodes in Graphiti/FalkorDB...\n');
    const upserted = await graphitiIngest(
      collection, textChunks, documentId, title, source, agentId, accessLevel
    );

    const result = {
      status:        'ok',
      document_id:   documentId,
      title,
      source,
      chunks:        textChunks.length,
      graphiti_episodes: upserted,
      collection,
      ..._imageMetadata && { image_metadata: _imageMetadata },
      ...storageKey && { storage: { bucket: S3_BUCKET, key: storageKey } },
    };
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  } catch (err) {
    process.stderr.write(JSON.stringify({ error: err.message }) + '\n');
    process.exit(1);
  }
})();
