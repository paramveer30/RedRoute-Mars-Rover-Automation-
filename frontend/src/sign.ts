// HMAC-SHA256 command signing. Mirrors backend/app/link.py canonical() + sign() exactly:
//   sig = hex(hmac_sha256(key, canonical(cmd) + String(seq)))
// canonical = JSON with sorted keys, no whitespace, nulls/undefined and "sig" dropped.

export function canonical(v: unknown): string {
  if (v === null || typeof v !== 'object') return JSON.stringify(v)
  if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`
  const obj = v as Record<string, unknown>
  const keys = Object.keys(obj)
    .filter((k) => k !== 'sig' && obj[k] !== undefined && obj[k] !== null)
    .sort()
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonical(obj[k])}`).join(',')}}`
}

export async function hmacHex(key: string, msg: string): Promise<string> {
  const enc = new TextEncoder()
  const k = await crypto.subtle.importKey('raw', enc.encode(key), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign'])
  const sig = await crypto.subtle.sign('HMAC', k, enc.encode(msg))
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

let lastSeq = 0

/** Strictly increasing across reloads (ms clock), as the rover rejects old seqs. */
export function nextSeq(): number {
  lastSeq = Math.max(Date.now(), lastSeq + 1)
  return lastSeq
}

export async function signCommand(key: string, cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
  const body: Record<string, unknown> = { ...cmd, seq: nextSeq() }
  for (const k of Object.keys(body)) if (body[k] === undefined || body[k] === null) delete body[k]
  body.sig = await hmacHex(key, canonical(body) + String(body.seq))
  return body
}
