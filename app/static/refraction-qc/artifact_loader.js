import { artifactRows, filterArtifactRows } from './filters.js';

export { artifactRows, filterArtifactRows };

export function parseCsvArtifact(text) {
  const rows = [];
  const lines = String(text || '').split(/\r?\n/).filter((line) => line.length);
  if (!lines.length) return { columns: [], records: [] };
  const columns = splitCsvLine(lines[0]);
  for (const line of lines.slice(1)) {
    const values = splitCsvLine(line);
    const record = {};
    columns.forEach((column, index) => {
      record[column] = values[index] ?? '';
    });
    rows.push(record);
  }
  return { columns, records: rows };
}

export function parseJsonArtifact(text) {
  const payload = JSON.parse(text);
  if (Array.isArray(payload)) {
    const columns = Array.from(new Set(payload.flatMap((record) => Object.keys(record || {}))));
    return { columns, records: payload };
  }
  return payload;
}

function splitCsvLine(line) {
  const values = [];
  let value = '';
  let inQuotes = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"' && line[index + 1] === '"') {
      value += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      values.push(value);
      value = '';
    } else {
      value += char;
    }
  }
  values.push(value);
  return values;
}
