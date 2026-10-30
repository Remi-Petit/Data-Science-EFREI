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
	<span class="label">{label}</span>
	<span class="track-wrap" style="--pct: {percent}%">
		<input type="range" bind:value {min} {max} step="1" />
		<span class="value">{value}</span>
	</span>
	<span class="bounds"><span>{min}</span><span>{max}</span></span>
</label>

<style>
	.field {
		display: block;
		margin-bottom: 1rem;
	}
	.label {
		display: block;
		font-size: 0.875rem;
		margin-bottom: 0.25rem;
	}
	.track-wrap {
		position: relative;
		display: block;
		padding-top: 1.5rem;
	}
	input {
		width: 100%;
		margin: 0;
		appearance: none;
		-webkit-appearance: none;
		height: 0.25rem;
		border-radius: 0.25rem;
		background: linear-gradient(to right, var(--primary) var(--pct), rgba(151, 166, 195, 0.25) var(--pct));
		outline: none;
		cursor: pointer;
	}
	input::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 0.75rem;
		height: 0.75rem;
		border-radius: 50%;
		background: var(--primary);
	}
	input::-moz-range-thumb {
		width: 0.75rem;
		height: 0.75rem;
		border: none;
		border-radius: 50%;
		background: var(--primary);
	}
	.value {
		position: absolute;
		top: 0;
		left: var(--pct);
		transform: translateX(-50%);
		color: var(--primary);
		font-size: 0.875rem;
	}
	.bounds {
		display: flex;
		justify-content: space-between;
		margin-top: 0.25rem;
		font-size: 0.75rem;
		color: var(--text-muted);
	}
</style>
