import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import fssync from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import FormData from 'form-data';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import os from 'os';
import {
  rotatePDFIfNeeded,
  convertFirstPageToImage,
  runFullPDFOCR,
  smartPDFProcess,
  optimizePDF,
  extractFirstPageAsPDF,
} from './utils/ocr.js';
import { logger } from './utils/logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const TEMP_DIR = path.join(__dirname, 'temp');
fssync.mkdirSync(TEMP_DIR, { recursive: true });
const upload = multer({ dest: TEMP_DIR });
app.use(express.json());

const TEMP_MAX_AGE_MS = Number(process.env.TEMP_MAX_AGE_HOURS ?? 24) * 60 * 60 * 1000;
const CALLBACK_TIMEOUT_MS = Number(process.env.CALLBACK_TIMEOUT_MS ?? 30000);

function isValidCallbackUrl(value) {
  if (!value || typeof value !== 'string') return false;
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

async function getFormHeadersWithLength(form) {
  const headers = form.getHeaders();
  try {
    // form-data v4: getLength ist callback-basiert, nicht Promise-basiert.
    const length = await new Promise((resolve, reject) => {
      form.getLength((err, len) => (err ? reject(err) : resolve(len)));
    });
    headers['Content-Length'] = length;
  } catch (err) {
    logger.warn(`Could not compute Content-Length, falling back to chunked: ${err.message}`);
  }
  return headers;
}

// Multer/busboy lesen den Dateinamen aus dem Content-Disposition-Header ohne
// Charset-Angabe als latin1 ein (busboy-Default: defParamCharset fehlt). Ein vom
// Client UTF-8-kodierter Name wie "Bestätigung (2).pdf" wird dadurch zu Mojibake
// ("BestaÌtigung (2).pdf"). Wir dekodieren die Bytes byteexakt zurück und
// normalisieren anschließend nach NFC (Umlaute kommen evtl. dekomponiert an).
function repairFilename(value) {
  let name = String(value ?? '');
  for (let pass = 0; pass < 2 && /[\u0080-\u00FF]/.test(name); pass += 1) {
    const decoded = Buffer.from(name, 'latin1').toString('utf8');
    // U+FFFD heißt: die Bytes waren gar kein gültiges UTF-8 → lieber behalten.
    if (decoded === name || decoded.includes('\uFFFD')) break;
    name = decoded;
  }
  return name.normalize('NFC');
}

function describeCallbackError(error) {
  if (error?.response) {
    return `status ${error.response.status} from ${error.config?.url}`;
  }
  if (error?.request && !error?.response) {
    return `no response from ${error.config?.url} (${error.code || error.message})`;
  }
  return error?.message || String(error);
}

async function postSuccessCallback(callbackUrl, { id, originalName, text, optimizedPath, firstPagePdfPath, imagePath }) {
  const form = new FormData();
  form.append('id', id);
  form.append('text', text ?? '');
  form.append('filename', originalName || `${id}.pdf`);
  // knownLength mitgeben, damit form.getLength() eine Content-Length
  // berechnen kann (ohne hängt der POST von Chunked-Encoding ab).
  const [pdfStat, page1Stat, imgStat] = await Promise.all([
    fs.stat(optimizedPath),
    fs.stat(firstPagePdfPath),
    fs.stat(imagePath),
  ]);
  form.append('pdf', fssync.createReadStream(optimizedPath), {
    filename: `optimized-${id}.pdf`,
    contentType: 'application/pdf',
    knownLength: pdfStat.size,
  });
  form.append('pdf_page1', fssync.createReadStream(firstPagePdfPath), {
    filename: `page1-${id}.pdf`,
    contentType: 'application/pdf',
    knownLength: page1Stat.size,
  });
  form.append('image', fssync.createReadStream(imagePath), {
    filename: `page1-${id}.png`,
    contentType: 'image/png',
    knownLength: imgStat.size,
  });

  const headers = await getFormHeadersWithLength(form);
  const response = await axios.post(callbackUrl, form, {
    headers,
    timeout: CALLBACK_TIMEOUT_MS,
    maxBodyLength: Infinity,
    maxContentLength: Infinity,
  });
  return response;
}

async function postErrorCallback(callbackUrl, payload) {
  await axios.post(callbackUrl, payload, {
    timeout: CALLBACK_TIMEOUT_MS,
    maxBodyLength: Infinity,
    maxContentLength: Infinity,
  });
}

async function removeFilesQuietly(files) {
  await Promise.allSettled(
    (files || []).filter(Boolean).map(async (file) => {
      try {
        await fs.unlink(file);
      } catch (err) {
        if (err?.code !== 'ENOENT') logger.warn(`Cleanup failed for ${file}: ${err.message}`);
      }
    })
  );
}

// Löscht alle Artefakte eines Jobs (alle Dateien mit id-Prefix) plus Extra-Dateien
// (z.B. der Multer-Upload, dessen Name nicht mit der Job-ID beginnt).
async function cleanupJobFiles(jobId, extraFiles = []) {
  try {
    const entries = await fs.readdir(TEMP_DIR);
    const jobFiles = entries
      .filter((name) => name.startsWith(jobId))
      .map((name) => path.join(TEMP_DIR, name));
    const files = [...jobFiles, ...extraFiles.filter(Boolean)];
    if (files.length === 0) return;
    logger.info(`Cleaning up ${files.length} temp file(s) for job ${jobId}...`);
    await removeFilesQuietly(files);
  } catch (err) {
    logger.warn(`Job cleanup failed for ${jobId}: ${err.message}`);
  }
}

// Entfernt verwaiste Dateien (z.B. nach Crash ohne finally-Cleanup) anhand des mtime-Alters.
async function cleanupStaleTempFiles() {
  try {
    const entries = await fs.readdir(TEMP_DIR);
    const now = Date.now();
    const stale = [];
    for (const name of entries) {
      const full = path.join(TEMP_DIR, name);
      try {
        const stat = await fs.stat(full);
        if (stat.isFile() && now - stat.mtimeMs > TEMP_MAX_AGE_MS) stale.push(full);
      } catch {
        // Datei zwischenzeitlich gelöscht → ignorieren
      }
    }
    if (stale.length > 0) {
      logger.info(`Removing ${stale.length} stale temp file(s)...`);
      await removeFilesQuietly(stale);
    }
  } catch (err) {
    logger.warn(`Stale temp cleanup failed: ${err.message}`);
  }
}

app.get('/', (req, res) => {
  const hostname = os.hostname();
  logger.info(`OCR microservice running on ${hostname}`);
  res.send(`OCR microservice running on ${hostname}`);
});

app.post('/ocr', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  const callbackUrl = req.body.callbackUrl;
  const id = uuidv4();
  const originalName = repairFilename(req.file.originalname);
  const tempDir = TEMP_DIR;
  const originalPath = path.join(tempDir, req.file.filename);
  const rotatedPdfPath = path.join(tempDir, `${id}-rotated.pdf`);
  const imagePrefix = path.join(tempDir, id);

  const hostname = os.hostname();
  logger.info(`OCR microservice running on ${hostname}`);
  logger.info('Received OCR job:', originalName);
  res.status(202).json({ message: 'OCR started', id, hostname });

  const hasCallback = isValidCallbackUrl(callbackUrl);
  if (callbackUrl && !hasCallback) {
    logger.warn(`Invalid callbackUrl ignored: ${callbackUrl}`);
  }

  try {
    const correctedPdf = await rotatePDFIfNeeded(originalPath, rotatedPdfPath);
    const imagePath = await convertFirstPageToImage(correctedPdf, imagePrefix);
    //const ocrResult = await runFullPDFOCR(correctedPdf, path.join(tempDir, id));
    const ocrResult = await smartPDFProcess(correctedPdf, path.join(tempDir, id));
    const optimizedPath = path.join(tempDir, `${id}-optimized.pdf`);
    await optimizePDF(ocrResult.pdf, optimizedPath);
    const firstPagePdfPath = path.join(tempDir, `${id}-page1.pdf`);
    await extractFirstPageAsPDF(optimizedPath, firstPagePdfPath);

    logger.info('OCR processing complete');

    // Erfolg und Zustellung getrennt behandeln: Ein fehlgeschlagener Callback
    // darf nicht als OCR-Fehler gemeldet werden.
    if (!hasCallback) {
      logger.warn('No valid callbackUrl provided — result not sent.');
      return;
    }

    try {
      logger.info('Sending OCR result as multipart to callback:', callbackUrl);
      const response = await postSuccessCallback(callbackUrl, {
        id,
        originalName,
        text: ocrResult.text,
        optimizedPath,
        firstPagePdfPath,
        imagePath,
      });
      logger.info(`Callback delivered (status ${response.status})`);
    } catch (callbackError) {
      // Kein zweiter POST an die gleiche defekte URL mit "OCR failed" —
      // die OCR war erfolgreich, nur die Zustellung ist gescheitert.
      logger.error('Success callback failed:', describeCallbackError(callbackError));
    }
  } catch (error) {
    logger.error('OCR failed:', error.message);
    if (hasCallback) {
      try {
        await postErrorCallback(callbackUrl, {
          id,
          error: 'OCR failed',
          message: error.message,
          filename: originalName || `${id}.pdf`
        });
        logger.info('Error callback delivered');
      } catch (callbackError) {
        logger.error('Error callback failed:', describeCallbackError(callbackError));
      }
    }
  } finally {
    await cleanupJobFiles(id, [originalPath]);
  }
});

app.listen(3000, async () => {
  logger.info('OCR microservice listening on port 3000');
  const hostname = os.hostname();
  logger.info(`OCR microservice running on ${hostname}`);
  await cleanupStaleTempFiles();
  setInterval(cleanupStaleTempFiles, 60 * 60 * 1000).unref();
});
