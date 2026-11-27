import express from 'express';
import os from 'os';
import { DateTime } from 'luxon';
import ical from 'ical-generator';
import { v4 as uuidv4 } from 'uuid';
import * as Sentry from '@sentry/node';

// --- GlitchTip (Sentry-kompat): Errors + Traces -----------------------------
// Nur aktiv, wenn SENTRY_DSN gesetzt ist (siehe .env.example / docker-compose).
// Ohne DSN verhält sich der Service wie bisher.
if (process.env.SENTRY_DSN) {
  Sentry.init({
    dsn: process.env.SENTRY_DSN,
    environment: process.env.SENTRY_ENVIRONMENT || 'dev',
    release: process.env.SENTRY_RELEASE || undefined,
    tracesSampleRate: Number(process.env.SENTRY_TRACES_SAMPLE_RATE ?? 0.1),
  });
  Sentry.setTag('service', 'ics-service');
}

const app = express();
const PORT = process.env.PORT || 3000;

// Obergrenzen gegen Missbrauch (Payloads sind rein textbasiert).
const MAX_EVENTS = Number(process.env.MAX_EVENTS ?? 2000);
const MAX_BIRTHDAYS = Number(process.env.MAX_BIRTHDAYS ?? 2000);

app.use(express.json({ limit: '512kb' }));

// --- Hilfsfunktionen --------------------------------------------------------

function badRequest(res, message) {
  return res.status(400).json({ error: message });
}

function isValidTimezone(zone) {
  if (typeof zone !== 'string' || zone.length === 0) return false;
  return DateTime.now().setZone(zone).isValid;
}

function isValidDate(value) {
  return typeof value === 'string' && DateTime.fromISO(value).isValid;
}

function sendIcs(res, fileName, calendar) {
  const icsContent = calendar.toString();
  const base64 = Buffer.from(icsContent, 'utf8').toString('base64');
  res.json({
    fileName,
    mimeType: 'text/calendar',
    data: base64,
  });
}

// --- Health-Check analog zu den anderen apiservices (tests/test_services.py)
app.get('/', (req, res) => {
  res.send(`ICS generator running on ${os.hostname()}`);
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'ics',
    hostname: os.hostname(),
    datetime: new Date().toISOString(),
  });
});

// --- POST /v1/generate-ics: generischer Termin-/Zeitplan-Kalender -----------
app.post('/v1/generate-ics', (req, res) => {
  console.log('generate-ics: new request');

  const body = req.body ?? {};
  const {
    timezone = 'Europe/Berlin',
    events = [],
    prodId = '//mybister.com//ics-generator//EN',
    name = 'myb Generic Calendar Name',
    calName = 'myb Generic Calendar Name',
  } = body;

  if (!isValidTimezone(timezone)) {
    return badRequest(res, `invalid timezone: ${JSON.stringify(timezone)}`);
  }
  if (!Array.isArray(events)) {
    return badRequest(res, 'events must be an array');
  }
  if (events.length > MAX_EVENTS) {
    return badRequest(res, `too many events (max ${MAX_EVENTS})`);
  }

  const calendar = ical({ prodId, name, timezone });
  calendar.name(calName);

  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (!ev || typeof ev !== 'object') {
      return badRequest(res, `events[${i}] must be an object`);
    }
    if (typeof ev.summary !== 'string' || ev.summary.trim() === '') {
      return badRequest(res, `events[${i}].summary is required`);
    }
    if (!isValidDate(ev.start)) {
      return badRequest(res, `events[${i}].start must be an ISO-8601 date-time`);
    }
    if (!isValidDate(ev.end)) {
      return badRequest(res, `events[${i}].end must be an ISO-8601 date-time`);
    }

    const start = DateTime.fromISO(ev.start, { zone: timezone });
    const end = DateTime.fromISO(ev.end, { zone: timezone });
    if (end.toMillis() <= start.toMillis()) {
      return badRequest(res, `events[${i}].end must be after start`);
    }

    const uid =
      ev.uid || `${start.toFormat("yyyyMMdd'T'HHmmss")}-${ev.summary}-${uuidv4()}`;

    calendar.createEvent({
      id: uid,
      start,
      end,
      summary: ev.summary,
      description: typeof ev.description === 'string' ? ev.description : undefined,
      location: typeof ev.location === 'string' ? ev.location : undefined,
      timezone,
    });
  }

  console.log(`generate-ics: ${events.length} event(s), timezone=${timezone}`);
  return sendIcs(res, 'schedule.ics', calendar);
});

// --- POST /v1/generate-birthday-cal: Jahresgeburtstagskalender ---------------
app.post('/v1/generate-birthday-cal', (req, res) => {
  console.log('generate-birthday-cal: new request');

  const body = req.body ?? {};
  const {
    birthdays = [],
    prodId = '//mybister.com//birthday-cal//EN',
    name = 'Birthday Calendar',
    calName = 'Birthday Calendar',
  } = body;

  if (!Array.isArray(birthdays)) {
    return badRequest(res, 'birthdays must be an array');
  }
  if (birthdays.length > MAX_BIRTHDAYS) {
    return badRequest(res, `too many birthdays (max ${MAX_BIRTHDAYS})`);
  }

  const calendar = ical({ prodId, name });
  calendar.name(calName);

  for (let i = 0; i < birthdays.length; i++) {
    const bd = birthdays[i];
    if (!bd || typeof bd !== 'object') {
      return badRequest(res, `birthdays[${i}] must be an object`);
    }
    if (typeof bd.name !== 'string' || bd.name.trim() === '') {
      return badRequest(res, `birthdays[${i}].name is required`);
    }
    if (!isValidDate(bd.date)) {
      return badRequest(res, `birthdays[${i}].date must be an ISO-8601 date (YYYY-MM-DD)`);
    }

    const date = DateTime.fromISO(bd.date);
    const uid = `${date.toFormat('MMdd')}-${bd.name}-${uuidv4()}`;

    calendar.createEvent({
      id: uid,
      start: date,
      allDay: true,
      summary: `Birthday: ${bd.name}`,
      repeating: { freq: 'YEARLY' },
    });
  }

  console.log(`generate-birthday-cal: ${birthdays.length} birthday(s)`);
  return sendIcs(res, 'birthdays.ics', calendar);
});

// --- Fehlerbehandlung -------------------------------------------------------
// Expliziter JSON-400 für kaputte Bodies (express.json wirft einen SyntaxError,
// der ohne diesen Handler als HTML-400 zurückgeht); alles andere wird als
// JSON-500 beantwortet und – falls konfiguriert – an GlitchTip gemeldet.
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  if (err?.type === 'entity.parse.failed' || err instanceof SyntaxError) {
    return res.status(400).json({ error: 'invalid JSON body' });
  }
  console.error('unhandled error:', err);
  if (process.env.SENTRY_DSN) {
    Sentry.captureException(err);
  }
  return res.status(500).json({ error: 'internal server error' });
});

app.listen(PORT, () => {
  console.log(`ICS Generator running on port ${PORT}`);
});
