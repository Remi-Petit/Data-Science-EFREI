<script lang="ts">
	let {
		label,
		value = $bindable(),
		min,
		max
	}: { label: string; value: number; min: number; max: number } = $props();

	const percent = $derived(((value - min) / (max - min)) * 100);
</script>

<label class="field">
	<span class="head">
		<span class="label">{label}</span>
		<span class="value">{value}</span>
	</span>
	<input type="range" bind:value {min} {max} step="1" style="--pct: {percent}%" />
	<span class="bounds"><span>{min}</span><span>{max}</span></span>
</label>

<style>
	.field {
		display: block;
		min-width: 0;
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		margin-bottom: 0.5rem;
	}
	.label {
		font-size: 0.82rem;
		font-weight: 550;
		color: var(--text-muted);
	}
	.value {
		padding: 0.05rem 0.55rem;
		border-radius: 999px;
		background: var(--primary-soft);
		color: var(--primary);
		font-size: 0.82rem;
		font-weight: 700;
	}
	input {
		width: 100%;
		margin: 0;
		appearance: none;
		-webkit-appearance: none;
		height: 0.4rem;
		border-radius: 999px;
		background: linear-gradient(to right, var(--primary) var(--pct), var(--bg-secondary) var(--pct));
		cursor: pointer;
	}
	input::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 1.1rem;
		height: 1.1rem;
		border-radius: 50%;
		background: #fff;
		border: 3px solid var(--primary);
		box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
	}
	input::-moz-range-thumb {
		width: 0.9rem;
		height: 0.9rem;
		border-radius: 50%;
		background: #fff;
		border: 3px solid var(--primary);
		box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
	}
	.bounds {
		display: flex;
		justify-content: space-between;
		margin-top: 0.3rem;
		font-size: 0.72rem;
		color: var(--text-muted);
	}
</style>
