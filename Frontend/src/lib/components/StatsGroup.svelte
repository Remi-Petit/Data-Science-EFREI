<script lang="ts">
	import { rankMedal, type Stats } from '$lib/api';
	import Alert from '$lib/components/Alert.svelte';
	import BarChart from '$lib/components/BarChart.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import Metric from '$lib/components/Metric.svelte';

	let {
		group,
		labels,
		rankCol,
		asPct = true
	}: { group: Stats; labels: Record<string, string>; rankCol: string; asPct?: boolean } = $props();

	const fmt = (v: number) => (asPct ? `${(v * 100).toFixed(2)}%` : v.toFixed(4));

	const sorted = $derived(
		Object.entries(group)
			.map(([key, metrics]) => ({ key, name: labels[key] ?? key, metrics }))
			.sort((a, b) => (b.metrics[rankCol] ?? 0) - (a.metrics[rankCol] ?? 0))
	);
	const metricCols = $derived(sorted.length ? Object.keys(sorted[0].metrics) : []);
</script>

{#if !sorted.length}
	<Alert kind="warning">Aucune statistique disponible pour ce groupe.</Alert>
{:else}
	<h4>🏆 Meilleur modèle : <strong>{sorted[0].name}</strong></h4>
	<div class="grid" style="grid-template-columns: repeat({metricCols.length}, minmax(0, 1fr))">
		{#each metricCols as c (c)}
			<Metric label={c} value={fmt(sorted[0].metrics[c])} />
		{/each}
	</div>
	<hr />
	<DataTable
		columns={['Rang', 'Modèle', ...metricCols]}
		rows={sorted.map((m, i) => [rankMedal(i), m.name, ...metricCols.map((c) => fmt(m.metrics[c]))])}
	/>
	<BarChart bars={sorted.map((m) => ({ label: m.name, value: m.metrics[rankCol] ?? 0 }))} format={fmt} />
{/if}

<style>
	@media (max-width: 900px) {
		.grid[style] {
			grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
		}
	}
</style>
