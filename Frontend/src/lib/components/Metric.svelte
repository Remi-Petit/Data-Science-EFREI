<script lang="ts">
	let { label, value }: { label: string; value: string | number } = $props();
</script>

<div class="metric">
	<div class="label">{label}</div>
	<div class="value">{value}</div>
</div>

<style>
	.metric {
		padding: 0.25rem 0 0.75rem;
	}
	.label {
		font-size: 0.875rem;
		color: var(--text-muted);
	}
	.value {
		font-size: 2.25rem;
		line-height: 1.3;
		font-weight: 400;
	}
</style>
