import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { logger } from './logger.js';

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args);
    let stderr = '';
    proc.stderr.on('data', (data) => (stderr += data));
    proc.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Command failed: ${command} ${args.join(' ')}\n${stderr}`));
    });
  });
}

export async function extractTextFromPDF(pdfPath) {
  logger.info("Prüfe auf vorhandenen Text im PDF...");
  const proc = spawn('pdftotext', ['-layout', pdfPath, '-']);
  
  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    
    proc.stdout.on('data', (data) => (stdout += data));
    proc.stderr.on('data', (data) => (stderr += data));
    
    proc.on('close', (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`pdftotext failed: ${stderr}`));
      }
    });
  });
}

export function hasSignificantText(text) {
  // Entferne Leerzeichen, Tabs, Zeilenumbrüche und andere Whitespace-Zeichen
  const cleanText = text.replace(/\s+/g, '').trim();
  
  // Prüfe ob genügend alphanumerische Zeichen vorhanden sind
  const alphanumericCount = (cleanText.match(/[a-zA-Z0-9]/g) || []).length;
  
  logger.info(`Gefundene alphanumerische Zeichen: ${alphanumericCount}`);
  
  // Schwellenwert: mindestens 50 alphanumerische Zeichen für "signifikanten" Text
  return alphanumericCount >= 50;
}

export async function getRotation(filePath) {
  logger.info("Checking PDF rotation...");
  const { spawn } = await import('child_process');
  const proc = spawn('pdfinfo', [filePath]);
  const output = await new Promise((resolve, reject) => {
    let data = '';
    proc.stdout.on('data', chunk => (data += chunk));
    proc.on('close', code => {
      code === 0 ? resolve(data) : reject(new Error('pdfinfo failed'));
    });
  });
  const match = output.match(/Page rot:\s+(\d+)/);
  const rotation = match ? parseInt(match[1], 10) : 0;
  logger.info(`Detected rotation: ${rotation} degrees`);
  return rotation;
}

export async function rotatePDFIfNeeded(inputPath, outputPath) {
  const rotation = await getRotation(inputPath);
  if (rotation !== 0) {
    const correction = 360 - rotation;
    logger.info(`Rotating PDF by ${correction} degrees...`);
    await runCommand('gs', [
      '-o', outputPath,
      '-sDEVICE=pdfwrite',
      '-dAutoRotatePages=/None',
      '-c', `<</EndPage {0 eq {${correction} rotate} {}}>> setpagedevice`,
      '-f', inputPath
    ]);
    logger.info(`Rotated PDF saved to: ${outputPath}`);
    return outputPath;
  } else {
    logger.info("No rotation needed. Using original PDF.");
    return inputPath;
  }
}

export async function convertFirstPageToImage(pdfPath, outputPrefix) {
  logger.info("Extracting first page to PNG image...");
  const firstPageImagePrefix = `${outputPrefix}-firstpage`;
  await runCommand('pdftoppm', ['-f', '1', '-l', '1', '-png', pdfPath, firstPageImagePrefix]);
  const candidates = [
    `${firstPageImagePrefix}-1.png`,
    `${firstPageImagePrefix}-01.png`
  ];

  for (const filePath of candidates) {
    try {
      await fs.access(filePath);
      logger.info(`Image created at: ${filePath}`);
      return filePath;
    } catch {
      // Datei existiert nicht → weiter prüfen
    }
  }

  throw new Error("First page image was not created");
}

export async function extractFirstPageAsPDF(inputPath, outputPath) {
  logger.info("Extracting first page as a separate PDF...");
  await runCommand('gs', [
    '-sDEVICE=pdfwrite', '-dNOPAUSE', '-dBATCH', '-dSAFER',
    '-dFirstPage=1', '-dLastPage=1',
    `-sOutputFile=${outputPath}`, inputPath
  ]);
  logger.info(`First-page-only PDF saved to: ${outputPath}`);
}

export async function optimizePDF(inputPath, outputPath) {
  logger.info("Optimizing PDF using Ghostscript...");
  await runCommand('gs', [
    '-sDEVICE=pdfwrite',
    '-dCompatibilityLevel=1.4',
    '-dPDFSETTINGS=/ebook',
    '-dNOPAUSE', '-dQUIET', '-dBATCH',
    `-sOutputFile=${outputPath}`, inputPath
  ]);
  logger.info(`Optimized PDF saved to: ${outputPath}`);
}

export async function runFullPDFOCR(inputPdfPath, outputBasePath) {
  logger.info("Converting PDF to images (one per page)...");
  const imagePrefix = `${outputBasePath}-page`;
  await runCommand('pdftoppm', ['-png', inputPdfPath, imagePrefix]);
  logger.info("Image conversion done.");

  const dir = path.dirname(outputBasePath);
  const prefix = path.basename(imagePrefix);
  const files = (await fs.readdir(dir))
    .filter(f => f.startsWith(prefix) && f.endsWith('.png'))
    .map(f => path.join(dir, f));

  if (files.length === 0) {
    throw new Error("No images generated from PDF.");
  }

  const listFilePath = `${outputBasePath}-files.txt`;
  await fs.writeFile(listFilePath, files.join('\n'));

  const ocrPdfPath = `${outputBasePath}.pdf`;
  const ocrTxtPath = `${outputBasePath}.txt`;

  await runCommand('tesseract', [listFilePath, outputBasePath, '-l', 'deu+eng', 'pdf', 'txt']);
  const text = await fs.readFile(ocrTxtPath, 'utf8');

  logger.info(`OCR text saved to: ${ocrTxtPath}`);
  logger.info(`OCR PDF saved to: ${ocrPdfPath}`);

  return {
    pdf: ocrPdfPath,
    text,
  };
}

// Neue Hauptfunktion die intelligente OCR-Erkennung durchführt
export async function smartPDFProcess(inputPdfPath, outputBasePath) {
  try {
    logger.info("Starte intelligente PDF-Verarbeitung...");
    
    // Zuerst prüfen ob bereits maschinenlesbarer Text vorhanden ist
    const existingText = await extractTextFromPDF(inputPdfPath);
    
    if (hasSignificantText(existingText)) {
      logger.info("PDF enthält bereits maschinenlesbaren Text. OCR wird übersprungen.");
      
      // Text in Datei speichern
      const txtOutputPath = `${outputBasePath}.txt`;
      await fs.writeFile(txtOutputPath, existingText);
      logger.info(`Extrahierter Text gespeichert in: ${txtOutputPath}`);
      
      return {
        pdf: inputPdfPath, // Original PDF verwenden
        text: existingText,
        ocrPerformed: false
      };
    } else {
      logger.info("PDF enthält keinen oder zu wenig maschinenlesbaren Text. OCR wird durchgeführt...");
      
      // OCR durchführen
      const result = await runFullPDFOCR(inputPdfPath, outputBasePath);
      
      return {
        ...result,
        ocrPerformed: true
      };
    }
    
  } catch (error) {
    logger.error("Fehler beim Extrahieren des vorhandenen Textes:", error.message);
    logger.info("Fallback: OCR wird durchgeführt...");
    
    // Fallback auf OCR wenn Text-Extraktion fehlschlägt
    const result = await runFullPDFOCR(inputPdfPath, outputBasePath);
    
    return {
      ...result,
      ocrPerformed: true
    };
  }
}
