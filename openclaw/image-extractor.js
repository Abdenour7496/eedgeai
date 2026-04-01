#!/usr/bin/env node
/**
 * image-extractor.js — Vision-based text extraction for images and medical formats.
 *
 * Supported formats
 * -----------------
 *   Regular images  .jpg .jpeg .png .gif .bmp .webp .tiff .tif .avif
 *   DICOM           .dcm .dicom (and files with no extension whose preamble
 *                   contains the DICM magic at offset 128)
 *   NIfTI           .nii  .nii.gz
 *
 * Pipeline
 * --------
 *   1. Decode / decompress raw bytes
 *   2. Extract structured metadata (DICOM tags / NIfTI header)
 *   3. Render a representative 2-D frame → 8-bit PNG via sharp
 *      (middle frame for multi-slice DICOM; middle axial slice for NIfTI)
 *   4. Send PNG to a vision model (OpenAI gpt-4o default, or Anthropic Claude)
 *   5. Return { text, imageType, metadata } — text is what ingest.js chunks
 *
 * Env vars
 * --------
 *   OPENAI_API_KEY        required for OpenAI vision
 *   ANTHROPIC_API_KEY     required for Anthropic vision
 *   VISION_BACKEND        anthropic (default) | openai | ollama
 *   VISION_MODEL          gpt-4o (OpenAI default) | claude-3-5-sonnet-20241022 (Anthropic default)
 *   VISION_MAX_TOKENS     default 1200
 *   VISION_MAX_PX         max pixel dimension before resize, default 1024
 *   NO_VISION             set to "1" to skip vision call (metadata only)
 */

'use strict';

const https = require('https');
const http  = require('http');
const zlib  = require('zlib');
const path  = require('path');

// ── Config ────────────────────────────────────────────────────────────────────

const VISION_BACKEND   = process.env.VISION_BACKEND   || 'anthropic';
const VISION_MAX_TOKENS = parseInt(process.env.VISION_MAX_TOKENS || '1200', 10);
const VISION_MAX_PX    = parseInt(process.env.VISION_MAX_PX || '1024', 10);
const NO_VISION        = process.env.NO_VISION === '1';
const OPENAI_KEY       = process.env.OPENAI_API_KEY   || '';
const ANTHROPIC_KEY    = process.env.ANTHROPIC_API_KEY || '';
const OLLAMA_BASE_URL  = process.env.OLLAMA_BASE_URL  || 'http://ollama:11434';
const OLLAMA_VISION_MODEL = process.env.OLLAMA_VISION_MODEL || 'llava:7b';

const VISION_MODEL = process.env.VISION_MODEL || (
  VISION_BACKEND === 'anthropic' ? 'claude-sonnet-4-20250514'
    : VISION_BACKEND === 'ollama' ? OLLAMA_VISION_MODEL
    : 'gpt-4o'
);

// ── Extension sets ────────────────────────────────────────────────────────────

const REGULAR_IMAGE_EXTS = new Set([
  '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.avif',
]);
const DICOM_EXTS = new Set(['.dcm', '.dicom', '']);
const NIFTI_EXTS = new Set(['.nii', '.gz']);   // .gz catches .nii.gz

// ── Lazy deps ─────────────────────────────────────────────────────────────────

function requireSharp() {
  try { return require('sharp'); }
  catch { throw new Error('sharp not installed — run: npm install sharp'); }
}

function requireDicomParser() {
  try { return require('dicom-parser'); }
  catch { throw new Error('dicom-parser not installed — run: npm install dicom-parser'); }
}

function requireNiftiReader() {
  try { return require('nifti-reader-js'); }
  catch { throw new Error('nifti-reader-js not installed — run: npm install nifti-reader-js'); }
}

// ── Magic-byte detection ──────────────────────────────────────────────────────

function isDicom(buf) {
  // DICOM preamble is 128 bytes followed by 'DICM'
  return buf.length >= 132 &&
    buf[128] === 0x44 && buf[129] === 0x49 &&
    buf[130] === 0x43 && buf[131] === 0x4D;
}

function isGzip(buf) {
  return buf.length >= 2 && buf[0] === 0x1F && buf[1] === 0x8B;
}

// ── HTTPS POST helper ─────────────────────────────────────────────────────────

function httpsPost(hostname, path_, headers, body) {
  return new Promise((resolve, reject) => {
    const bodyBuf = Buffer.from(body);
    const req = https.request({
      hostname, path: path_, method: 'POST',
      headers: { ...headers, 'Content-Length': bodyBuf.length },
    }, res => {
      const chunks = [];
      res.on('data', d => chunks.push(d));
      res.on('end', () => {
        try {
          const parsed = JSON.parse(Buffer.concat(chunks).toString());
          if (parsed.error) return reject(new Error(
            parsed.error.message || JSON.stringify(parsed.error)
          ));
          resolve(parsed);
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.end(bodyBuf);
  });
}

// ── HTTP POST helper (for local services like Ollama) ────────────────────────

function httpPost(urlStr, headers, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlStr);
    const bodyBuf = Buffer.from(body);
    const req = http.request({
      hostname: url.hostname, port: url.port, path: url.pathname, method: 'POST',
      headers: { ...headers, 'Content-Length': bodyBuf.length },
    }, res => {
      const chunks = [];
      res.on('data', d => chunks.push(d));
      res.on('end', () => {
        try {
          const parsed = JSON.parse(Buffer.concat(chunks).toString());
          if (parsed.error) return reject(new Error(
            parsed.error.message || JSON.stringify(parsed.error)
          ));
          resolve(parsed);
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.end(bodyBuf);
  });
}

// ── Vision API calls ──────────────────────────────────────────────────────────

async function openaiVision(pngBase64, prompt) {
  if (!OPENAI_KEY) throw new Error('OPENAI_API_KEY not set');
  const body = JSON.stringify({
    model: VISION_MODEL,
    max_tokens: VISION_MAX_TOKENS,
    messages: [{
      role: 'user',
      content: [
        { type: 'image_url', image_url: { url: `data:image/png;base64,${pngBase64}`, detail: 'high' } },
        { type: 'text', text: prompt },
      ],
    }],
  });
  const res = await httpsPost('api.openai.com', '/v1/chat/completions', {
    Authorization: `Bearer ${OPENAI_KEY}`,
    'Content-Type': 'application/json',
  }, body);
  return res.choices[0].message.content.trim();
}

async function anthropicVision(pngBase64, prompt) {
  if (!ANTHROPIC_KEY) throw new Error('ANTHROPIC_API_KEY not set');
  const body = JSON.stringify({
    model: VISION_MODEL,
    max_tokens: VISION_MAX_TOKENS,
    messages: [{
      role: 'user',
      content: [
        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: pngBase64 } },
        { type: 'text', text: prompt },
      ],
    }],
  });
  const res = await httpsPost('api.anthropic.com', '/v1/messages', {
    'x-api-key':         ANTHROPIC_KEY,
    'anthropic-version': '2023-06-01',
    'Content-Type':      'application/json',
  }, body);
  return res.content[0].text.trim();
}

async function ollamaVision(pngBase64, prompt) {
  const body = JSON.stringify({
    model: VISION_MODEL,
    messages: [{
      role: 'user',
      content: [
        { type: 'image_url', image_url: { url: `data:image/png;base64,${pngBase64}` } },
        { type: 'text', text: prompt },
      ],
    }],
    max_tokens: VISION_MAX_TOKENS,
  });
  const res = await httpPost(`${OLLAMA_BASE_URL}/v1/chat/completions`, {
    'Content-Type': 'application/json',
  }, body);
  return res.choices[0].message.content.trim();
}

async function callVision(pngBase64, prompt) {
  if (VISION_BACKEND === 'anthropic') return anthropicVision(pngBase64, prompt);
  if (VISION_BACKEND === 'ollama')    return ollamaVision(pngBase64, prompt);
  return openaiVision(pngBase64, prompt);
}

// ── Sharp: resize + encode to base64 PNG ─────────────────────────────────────

async function toPngBase64(rawGrayscaleBuf, width, height) {
  const sharp = requireSharp();
  const resized = await sharp(rawGrayscaleBuf, {
    raw: { width, height, channels: 1 },
  })
    .resize({
      width:  Math.min(width,  VISION_MAX_PX),
      height: Math.min(height, VISION_MAX_PX),
      fit:    'inside',
    })
    .png()
    .toBuffer();
  return resized.toString('base64');
}

async function regularImageToPngBase64(rawBuf) {
  const sharp = requireSharp();
  const resized = await sharp(rawBuf)
    .resize({ width: VISION_MAX_PX, height: VISION_MAX_PX, fit: 'inside', withoutEnlargement: true })
    .png()
    .toBuffer();
  return resized.toString('base64');
}

// ── Pixel normalization (16-bit → 8-bit) ──────────────────────────────────────

function normalizeToUint8(pixelArray, windowCenter, windowWidth) {
  let low, high;

  if (windowCenter != null && windowWidth != null && windowWidth > 0) {
    low  = windowCenter - windowWidth / 2;
    high = windowCenter + windowWidth / 2;
  } else {
    // Auto window from data range
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < pixelArray.length; i++) {
      if (pixelArray[i] < min) min = pixelArray[i];
      if (pixelArray[i] > max) max = pixelArray[i];
    }
    low  = min;
    high = max;
    if (high === low) high = low + 1;
  }

  const range  = high - low;
  const output = new Uint8Array(pixelArray.length);
  for (let i = 0; i < pixelArray.length; i++) {
    const v = pixelArray[i];
    if      (v <= low)  output[i] = 0;
    else if (v >= high) output[i] = 255;
    else                output[i] = Math.round(255 * (v - low) / range);
  }
  return output;
}

// ── DICOM ─────────────────────────────────────────────────────────────────────

// DICOM transfer syntax UIDs that carry uncompressed pixel data
const UNCOMPRESSED_TS = new Set([
  '1.2.840.10008.1.2',    // Implicit VR Little Endian
  '1.2.840.10008.1.2.1',  // Explicit VR Little Endian
  '1.2.840.10008.1.2.2',  // Explicit VR Big Endian
]);

function dicomStr(ds, tag) {
  try { return ds.string(tag)?.trim() || ''; } catch { return ''; }
}

function dicomNum(ds, tag) {
  try {
    const s = ds.string(tag);
    const n = parseFloat(s);
    return isNaN(n) ? null : n;
  } catch { return null; }
}

async function extractDicom(buf) {
  const dicomParser = requireDicomParser();
  const byteArray   = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const ds          = dicomParser.parseDicom(byteArray);

  // ── Metadata ────────────────────────────────────────────────────────────────
  const modality        = dicomStr(ds, 'x00080060');
  const studyDesc       = dicomStr(ds, 'x00081030');
  const seriesDesc      = dicomStr(ds, 'x0008103e');
  const manufacturer    = dicomStr(ds, 'x00080070');
  const institution     = dicomStr(ds, 'x00080080');
  const protocolName    = dicomStr(ds, 'x00181030');
  const kvp             = dicomStr(ds, 'x00181060');
  const sliceThickness  = dicomStr(ds, 'x00180050');
  const pixelSpacing    = dicomStr(ds, 'x00280030');
  const rows            = dicomNum(ds, 'x00280010');
  const cols            = dicomNum(ds, 'x00280011');
  const bitsAllocated   = dicomNum(ds, 'x00280100') || 16;
  const bitsStored      = dicomNum(ds, 'x00280101') || bitsAllocated;
  const pixelRep        = dicomNum(ds, 'x00280103') || 0;  // 0=unsigned, 1=signed
  const numberOfFrames  = dicomNum(ds, 'x00280008') || 1;
  const photometric     = dicomStr(ds, 'x00280004');
  const windowCenter    = dicomNum(ds, 'x00281050');
  const windowWidth     = dicomNum(ds, 'x00281051');
  const transferSyntax  = dicomStr(ds, 'x00020010') || '1.2.840.10008.1.2.1';

  const metadata = {
    format: 'DICOM', modality, study_description: studyDesc,
    series_description: seriesDesc, institution, manufacturer,
    protocol_name: protocolName, kvp, slice_thickness: sliceThickness,
    pixel_spacing: pixelSpacing, rows, cols,
    bits_allocated: bitsAllocated, bits_stored: bitsStored,
    photometric_interpretation: photometric, number_of_frames: numberOfFrames,
    transfer_syntax: transferSyntax,
  };

  const metaText = [
    `[MEDICAL IMAGE: DICOM${modality ? ` / ${modality}` : ''}]`,
    '',
    '=== DICOM METADATA ===',
    modality       && `Modality            : ${modality}`,
    studyDesc      && `Study Description   : ${studyDesc}`,
    seriesDesc     && `Series Description  : ${seriesDesc}`,
    institution    && `Institution         : ${institution}`,
    manufacturer   && `Manufacturer        : ${manufacturer}`,
    protocolName   && `Protocol            : ${protocolName}`,
    kvp            && `KVP                 : ${kvp} kV`,
    sliceThickness && `Slice Thickness     : ${sliceThickness} mm`,
    pixelSpacing   && `Pixel Spacing       : ${pixelSpacing} mm`,
    rows && cols   && `Dimensions          : ${cols} × ${rows} px`,
    numberOfFrames > 1 && `Frames              : ${numberOfFrames}`,
    `Bit Depth           : ${bitsStored}-bit`,
    photometric    && `Photometric         : ${photometric}`,
  ].filter(Boolean).join('\n');

  // ── Pixel rendering ─────────────────────────────────────────────────────────
  const pixelEl = ds.elements.x7fe00010;
  let visionText = '';

  if (pixelEl && rows && cols && UNCOMPRESSED_TS.has(transferSyntax)) {
    try {
      const bytesPerPixel = bitsAllocated / 8;
      const frameSize     = rows * cols * bytesPerPixel;
      const frameIndex    = Math.floor(numberOfFrames / 2);   // middle frame
      const frameOffset   = pixelEl.dataOffset + frameIndex * frameSize;

      let pixelArray;
      if (bitsAllocated === 8) {
        pixelArray = pixelRep === 1
          ? new Int8Array(byteArray.buffer, frameOffset, rows * cols)
          : new Uint8Array(byteArray.buffer, frameOffset, rows * cols);
      } else {
        pixelArray = pixelRep === 1
          ? new Int16Array(byteArray.buffer, frameOffset, rows * cols)
          : new Uint16Array(byteArray.buffer, frameOffset, rows * cols);
      }

      const gray8 = normalizeToUint8(pixelArray, windowCenter, windowWidth);
      const b64   = await toPngBase64(Buffer.from(gray8), cols, rows);

      if (!NO_VISION) {
        const prompt = medicalVisionPrompt(modality || 'unknown');
        visionText = await callVision(b64, prompt);
      }
    } catch (e) {
      visionText = `[Pixel rendering failed: ${e.message}]`;
    }
  } else if (pixelEl && !UNCOMPRESSED_TS.has(transferSyntax)) {
    visionText = `[Compressed pixel data (transfer syntax: ${transferSyntax}) — metadata only]`;
  }

  const text = [
    metaText,
    visionText && '\n=== VISUAL ANALYSIS ===\n' + visionText,
  ].filter(Boolean).join('\n');

  return { text, imageType: 'dicom', metadata };
}

// ── NIfTI ─────────────────────────────────────────────────────────────────────

// NIfTI-1 datatype codes → typed array constructor
const NIFTI_DTYPES = {
  2:   Uint8Array,   // UINT8
  4:   Int16Array,   // INT16
  8:   Int32Array,   // INT32
  16:  Float32Array, // FLOAT32
  64:  Float64Array, // FLOAT64
  256: Int8Array,    // INT8
  512: Uint16Array,  // UINT16
  768: Uint32Array,  // UINT32
};

async function extractNifti(buf) {
  const nifti = requireNiftiReader();

  // Decompress if gzip
  let rawBuf = buf;
  if (isGzip(buf)) {
    rawBuf = Buffer.from(zlib.gunzipSync(buf));
  }

  const arrayBuf = rawBuf.buffer.slice(
    rawBuf.byteOffset, rawBuf.byteOffset + rawBuf.byteLength
  );

  if (!nifti.isNIFTI(arrayBuf)) {
    throw new Error('Buffer is not a valid NIfTI file');
  }

  const header = nifti.readHeader(arrayBuf);
  const imgBuf = nifti.readImage(header, arrayBuf);

  // Header fields
  const nx      = header.dims[1] || 1;
  const ny      = header.dims[2] || 1;
  const nz      = header.dims[3] || 1;
  const nt      = header.dims[4] || 1;
  const dx      = header.pixDims[1];
  const dy      = header.pixDims[2];
  const dz      = header.pixDims[3];
  const dtype   = header.datatypeCode;
  const intent  = header.intent_code;
  const descrip = header.description?.replace(/\0/g, '').trim() || '';

  const metadata = {
    format: 'NIfTI', dims: [nx, ny, nz, nt],
    voxel_size_mm: [dx, dy, dz],
    datatype_code: dtype, intent_code: intent,
    description: descrip,
  };

  const metaText = [
    '[MEDICAL IMAGE: NIfTI]',
    '',
    '=== NIfTI HEADER ===',
    `Dimensions          : ${nx} × ${ny} × ${nz}${nt > 1 ? ` × ${nt} (time)` : ''}`,
    `Voxel Size          : ${dx?.toFixed(3)} × ${dy?.toFixed(3)} × ${dz?.toFixed(3)} mm`,
    `Data Type           : ${dtype}`,
    intent  && `Intent Code         : ${intent}`,
    descrip && `Description         : ${descrip}`,
  ].filter(Boolean).join('\n');

  // Extract middle axial slice (z = nz/2)
  let visionText = '';
  try {
    const TypedArray = NIFTI_DTYPES[dtype];
    if (!TypedArray) throw new Error(`Unsupported NIfTI datatype: ${dtype}`);

    const sliceIndex = Math.floor(nz / 2);
    const slicePixels = nx * ny;
    const volumeArray = new TypedArray(imgBuf);
    const sliceOffset = sliceIndex * slicePixels;
    const sliceData   = volumeArray.slice(sliceOffset, sliceOffset + slicePixels);

    const gray8 = normalizeToUint8(sliceData, null, null);
    const b64   = await toPngBase64(Buffer.from(gray8), nx, ny);

    if (!NO_VISION) {
      const prompt = medicalVisionPrompt('MRI/NIfTI');
      visionText = await callVision(b64, prompt);
    }
  } catch (e) {
    visionText = `[Slice rendering failed: ${e.message}]`;
  }

  const text = [
    metaText,
    visionText && '\n=== VISUAL ANALYSIS ===\n' + visionText,
  ].filter(Boolean).join('\n');

  return { text, imageType: 'nifti', metadata };
}

// ── Regular images ────────────────────────────────────────────────────────────

async function extractRegularImage(buf, ext) {
  const metadata = { format: 'image', extension: ext };
  let visionText = '';

  if (!NO_VISION) {
    const b64    = await regularImageToPngBase64(buf);
    const prompt = `Describe this image in detail, including:
1. Main subject and overall content
2. Any text, labels, annotations, measurements, or overlays visible
3. Key visual elements, colors, patterns, or notable features
4. Context, setting, or apparent purpose of the image
Be thorough and objective.`;
    visionText = await callVision(b64, prompt);
  }

  const text = [
    `[IMAGE: ${ext.toUpperCase().replace('.', '') || 'UNKNOWN'}]`,
    '',
    visionText || '[Vision analysis skipped — NO_VISION=1]',
  ].join('\n');

  return { text, imageType: 'regular', metadata };
}

// ── Prompts ───────────────────────────────────────────────────────────────────

function medicalVisionPrompt(modality) {
  return `This is a medical image (${modality}). Describe in clinical detail:
1. Imaging modality and acquisition plane or orientation (if identifiable)
2. Anatomical region and structures visible
3. Notable findings, abnormalities, or pathologies (describe objectively; do not diagnose)
4. Image quality, contrast, and any artifacts
5. Any text, measurements, annotations, or overlays present in the image
Be precise, systematic, and use appropriate anatomical terminology.`;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * extractImageText(buffer, ext, filePath?)
 *
 * Returns { text, imageType, metadata }
 *   text      — descriptive string ready for chunking + embedding
 *   imageType — 'dicom' | 'nifti' | 'regular'
 *   metadata  — structured object with image-specific fields
 */
async function extractImageText(buf, ext, filePath = '') {
  const lext = ext.toLowerCase();

  // NIfTI: check extension first, then fall through
  if (lext === '.nii' || (lext === '.gz' && filePath.toLowerCase().endsWith('.nii.gz'))) {
    return extractNifti(buf);
  }

  // DICOM: explicit extension OR magic bytes
  if (DICOM_EXTS.has(lext) && isDicom(buf)) {
    return extractDicom(buf);
  }

  // Explicit .dcm/.dicom even if magic is absent (some writers omit preamble)
  if (lext === '.dcm' || lext === '.dicom') {
    return extractDicom(buf);
  }

  // Regular images
  if (REGULAR_IMAGE_EXTS.has(lext)) {
    return extractRegularImage(buf, lext);
  }

  throw new Error(`Unsupported image format: ${lext}`);
}

/** List of all extensions handled by this module */
const SUPPORTED_EXTENSIONS = [
  ...REGULAR_IMAGE_EXTS,
  '.dcm', '.dicom', '.nii', '.nii.gz',
];

module.exports = { extractImageText, SUPPORTED_EXTENSIONS, isDicom, isGzip };
