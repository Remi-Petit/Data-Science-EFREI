<script lang="ts">
	let { label, value }: { label: string; value: number } = $props();

	const tone = $derived(value < 0.33 ? 'success' : value < 0.66 ? 'warning' : 'error');
</script>

<div class="prob {tone}">
	<div class="head">
		<span class="label">{label}</span>
		<span class="pct">{(value * 100).toFixed(2)}%</span>
	</div>
	<div class="track"><div class="fill" style="width: {Math.max(value * 100, 1)}%"></div></div>
</div>

<style>
	.prob {
		--c: var(--success);
	}
	.warning {
		--c: var(--warning);
	}
	.error {
		--c: var(--error);
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		margin-bottom: 0.4rem;
	}
	.label {
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--text-muted);
	}
	.pct {
		font-size: 1.65rem;
		font-weight: 750;
		letter-spacing: -0.02em;
		color: var(--c);
	}
	.track {
		height: 0.55rem;
		border-radius: 999px;
		background: var(--bg-secondary);
		overflow: hidden;
	}
	.fill {
		height: 100%;
		border-radius: 999px;
		background: var(--c);
		transition: width 0.4s ease;
	}
</style>
