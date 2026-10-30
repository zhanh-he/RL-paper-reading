import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const filename = fileURLToPath(new URL('./runs/2026-10-30/sheet-dependency-failure.json', import.meta.url));
const receipt = JSON.parse(fs.readFileSync(filename, 'utf8'));
// The public failure summary needs the dependency error, not machine tracebacks.
for (const sample of receipt.samples) delete sample.traceback;
receipt.public_scope = 'Sanitized dependency-failure summary; raw machine tracebacks retained only with the external run.';
fs.writeFileSync(filename, JSON.stringify(receipt, null, 2) + '\n');
