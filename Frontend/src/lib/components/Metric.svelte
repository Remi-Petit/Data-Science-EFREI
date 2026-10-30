<script lang="ts">
	let { label, value, small = false }: { label: string; value: string | number; small?: boolean } = $props();
</script>

<div class="metric" class:small>
	<div class="label">{label}</div>
	<div class="value">{value}</div>
</div>

<style>
	.metric {
		padding: 0.9rem 1rem;
		background: var(--bg-secondary);
		border-radius: var(--radius);
		min-width: 0;
	}
	.label {
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--text-muted);
	}
	.value {
		margin-top: 0.15rem;
		font-size: 1.65rem;
		font-weight: 700;
		letter-spacing: -0.02em;
		line-height: 1.3;
		overflow-wrap: anywhere;
	}
	.small .value {
		font-size: 1.15rem;
	}
</style>
