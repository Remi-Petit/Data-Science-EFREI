const API = '/api';

export async function apiGet<T>(path: string): Promise<T> {
	const r = await fetch(`${API}${path}`);
	if (!r.ok) throw new Error(`HTTP ${r.status}`);
	return r.json();
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
	const r = await fetch(`${API}${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!r.ok) {
		let detail = `HTTP ${r.status}`;
		try {
			const j = await r.json();
			if (j?.detail) detail = String(j.detail);
		} catch {
			/* corps non JSON */
		}
		throw new Error(detail);
	}
	return r.json();
}

export type ModelEntry = { name: string; label: string };
export type Stats = Record<string, Record<string, number>>;

const MEDALS = ['🥇', '🥈', '🥉'];

export function modelLabel(key: string, ordered: string[], labels: Record<string, string>): string {
	const idx = ordered.indexOf(key);
	const base = labels[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
	return idx >= 0 && idx < MEDALS.length ? `${MEDALS[idx]} ${base}` : base;
}

export function rankMedal(i: number): string {
	return i < MEDALS.length ? MEDALS[i] : `#${i + 1}`;
}

export const pct = (v: number, digits = 2) => `${(v * 100).toFixed(digits)}%`;
