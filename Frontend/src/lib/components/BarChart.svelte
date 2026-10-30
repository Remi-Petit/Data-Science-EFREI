<script lang="ts">
	type Bar = { label: string; value: number };

	let { bars, format }: { bars: Bar[]; format: (v: number) => string } = $props();

	const max = $derived(Math.max(...bars.map((b) => b.value), 0) || 1);
</script>

<div class="chart">
	{#each bars as b (b.label)}
		<div class="row">
			<span class="name">{b.label}</span>
			<span class="track">
				<span class="bar" style="width: {Math.max((b.value / max) * 100, 0)}%"></span>
			</span>
			<span class="val">{format(b.value)}</span>
		</div>
	{/each}
</div>

<style>
	.chart {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		padding: 0.5rem 0;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(6rem, 12rem) 1fr 5rem;
		align-items: center;
		gap: 0.75rem;
	}
	.name {
		text-align: right;
		font-size: 0.875rem;
	}
	.track {
		height: 1.75rem;
		background-image: linear-gradient(to right, rgba(49, 51, 63, 0.08) 1px, transparent 1px);
		background-size: 25% 100%;
		border-left: 1px solid rgba(49, 51, 63, 0.08);
	}
	.bar {
		display: block;
		height: 100%;
		background: var(--chart);
		border-radius: 0 2px 2px 0;
		transition: width 0.3s;
	}
	.val {
		font-size: 0.875rem;
	}
	@media (max-width: 600px) {
		.row {
			grid-template-columns: 6rem 1fr 4rem;
		}
	}
</style>
