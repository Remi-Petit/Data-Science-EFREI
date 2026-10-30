<script lang="ts">
	import { rankMedal, type Stats } from '$lib/api';
	import Alert from '$lib/components/Alert.svelte';
	import BarChart from '$lib/components/BarChart.svelte';
	import Card from '$lib/components/Card.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import Icon from '$lib/components/Icon.svelte';
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
	<div class="stack">
		<div class="best">
			<span class="trophy"><Icon name="trophy" size={22} /></span>
			<div>
				<div class="caption">Meilleur modèle</div>
				<strong>{sorted[0].name}</strong>
			</div>
		</div>

		<div class="grid metrics">
			{#each metricCols as c (c)}
				<Metric label={c} value={fmt(sorted[0].metrics[c])} />
			{/each}
		</div>

		<Card title="Classement" icon="layers">
			<DataTable
				columns={['Rang', 'Modèle', ...metricCols]}
				rows={sorted.map((m, i) => [rankMedal(i), m.name, ...metricCols.map((c) => fmt(m.metrics[c]))])}
			/>
		</Card>

		<Card title={rankCol} icon="bar-chart">
			<BarChart bars={sorted.map((m) => ({ label: m.name, value: m.metrics[rankCol] ?? 0 }))} format={fmt} />
		</Card>
	</div>
{/if}

<style>
	.best {
		display: flex;
		align-items: center;
		gap: 0.9rem;
		padding: 1rem 1.25rem;
		border-radius: var(--radius-lg);
		background: var(--gradient);
		color: #fff;
	}
	.best .caption {
		color: rgba(255, 255, 255, 0.8);
	}
	.best strong {
		font-size: 1.35rem;
	}
	.trophy {
		display: grid;
		place-items: center;
		width: 2.8rem;
		height: 2.8rem;
		border-radius: 0.8rem;
		background: rgba(255, 255, 255, 0.2);
	}
	.metrics {
		grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
		gap: 0.9rem;
	}
</style>
