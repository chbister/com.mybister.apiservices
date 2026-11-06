import { format } from 'util';

function timestamp(date = new Date()) {
  const pad = (n) => String(n).padStart(2, '0');
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
  );
}

function write(severity, args, sink) {
  sink(`${timestamp()} [${severity}] ${format(...args)}`);
}

export const logger = {
  info: (...args) => write('INFO', args, console.log),
  warn: (...args) => write('WARN', args, console.warn),
  error: (...args) => write('ERROR', args, console.error),
};
