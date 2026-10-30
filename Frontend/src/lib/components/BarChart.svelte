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
				<span class="bar" style="width: {Math.max((b.value / max) * 100, 1)}%"></span>
			</span>
			<span class="val">{format(b.value)}</span>
		</div>
	{/each}
</div>

<style>
	.chart {
		display: flex;
		flex-direction: column;
		gap: 0.7rem;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(6rem, 11rem) 1fr 4.5rem;
		align-items: center;
		gap: 0.9rem;
	}
	.name {
		text-align: right;
		font-size: 0.85rem;
		font-weight: 550;
	}
	.track {
		height: 1.5rem;
		border-radius: 999px;
		background: var(--bg-secondary);
		overflow: hidden;
	}
	.bar {
		display: block;
		height: 100%;
		border-radius: 999px;
		background: var(--gradient);
		transition: width 0.4s ease;
	}
	.val {
		font-size: 0.85rem;
		font-weight: 650;
	}
	@media (max-width: 640px) {
		.row {
			grid-template-columns: 5.5rem 1fr 4rem;
			gap: 0.5rem;
		}
	}
</style>
