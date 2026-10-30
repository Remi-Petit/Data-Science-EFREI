<script lang="ts">
	let {
		label,
		value = $bindable(),
		step = 1,
		min,
		max,
		decimals
	}: { label: string; value: number; step?: number; min?: number; max?: number; decimals?: number } = $props();

	const clamp = (v: number) => Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v));

	function nudge(dir: 1 | -1) {
		const next = (Number(value) || 0) + dir * step;
		value = clamp(Number(next.toFixed(6)));
	}

	function onchange(e: Event) {
		const n = (e.currentTarget as HTMLInputElement).valueAsNumber;
		value = Number.isNaN(n) ? (min ?? 0) : clamp(n);
	}

	const shown = $derived(decimals !== undefined ? Number(value).toFixed(decimals) : value);
</script>

<label class="field">
	<span class="label">{label}</span>
	<span class="box">
		<input type="number" value={shown} {step} {min} {max} {onchange} />
		<button type="button" tabindex="-1" aria-label="Diminuer" onclick={() => nudge(-1)}>−</button>
		<button type="button" tabindex="-1" aria-label="Augmenter" onclick={() => nudge(1)}>+</button>
	</span>
</label>

<style>
	.field {
		display: block;
		min-width: 0;
	}
	.label {
		display: block;
		font-size: 0.82rem;
		font-weight: 550;
		color: var(--text-muted);
		margin-bottom: 0.35rem;
	}
	.box {
		display: flex;
		align-items: stretch;
		background: var(--bg-secondary);
		border: 1px solid transparent;
		border-radius: var(--radius);
		overflow: hidden;
		transition: border-color 0.15s, box-shadow 0.15s;
	}
	.box:focus-within {
		border-color: var(--primary);
		box-shadow: 0 0 0 3px var(--primary-soft);
	}
	input {
		flex: 1;
		min-width: 0;
		border: none;
		background: transparent;
		padding: 0.55rem 0.8rem;
		outline: none;
		appearance: textfield;
		-moz-appearance: textfield;
	}
	input::-webkit-outer-spin-button,
	input::-webkit-inner-spin-button {
		-webkit-appearance: none;
		margin: 0;
	}
	button {
		width: 2.1rem;
		border: none;
		border-left: 1px solid var(--border);
		background: transparent;
		color: var(--text-muted);
		cursor: pointer;
		font-size: 1.1rem;
	}
	button:hover {
		color: var(--primary);
		background: var(--primary-soft);
	}
</style>
