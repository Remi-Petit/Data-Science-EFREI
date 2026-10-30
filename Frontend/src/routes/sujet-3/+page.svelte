<script lang="ts">
	import { onMount } from 'svelte';
	import { apiGet, apiPost, modelLabel, type ModelEntry, type Stats } from '$lib/api';
	import Alert from '$lib/components/Alert.svelte';
	import BarChart from '$lib/components/BarChart.svelte';
	import Card from '$lib/components/Card.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Metric from '$lib/components/Metric.svelte';
	import MultiSelect from '$lib/components/MultiSelect.svelte';
	import NumberInput from '$lib/components/NumberInput.svelte';
	import PageHeader from '$lib/components/PageHeader.svelte';
	import SelectInput from '$lib/components/SelectInput.svelte';

	type Perf = 'Low' | 'Medium' | 'High';
	type Result = { sales_prediction: number; performance: Perf; roi_estimate: number | null };

	const PERF = { Low: 'error', Medium: 'warning', High: 'success' } as const;

	const FALLBACK = ['linear_regression', 'random_forest', 'xgboost', 'mlp'];
	let available = $state<string[]>(FALLBACK);
	let labels = $state<Record<string, string>>({});
	let selected = $state<string[]>(['linear_regression']);

	onMount(async () => {
		try {
			const { models } = await apiGet<{ models: ModelEntry[] }>('/sujet-3/models');
			const names = models.map((e) => e.name);
			labels = Object.fromEntries(models.map((e) => [e.name, e.label]));
			try {
				const { stats } = await apiGet<{ stats: Record<string, Stats> }>('/sujet-3/stats');
				const g = stats.regression ?? {};
				names.sort((a, b) => (g[b]?.['R²'] ?? 0) - (g[a]?.['R²'] ?? 0));
			} catch {
				/* ordre de l'API */
			}
			available = names;
			selected = names.slice(0, 1);
		} catch {
			/* valeurs de repli */
		}
	});

	const opts = $derived(available.map((k) => ({ value: k, label: modelLabel(k, available, labels) })));
	const lbl = (k: string) => modelLabel(k, available, labels);

	let tv = $state(50.0);
	let radio = $state(18.0);
	let social_media = $state(3.0);
	let influencer = $state('Macro');

	const total = $derived(tv + radio + social_media);
	const share = (v: number) => (total > 0 ? `${((v / total) * 100).toFixed(1)} %` : '–');

	let results = $state<Record<string, Result> | null>(null);
	let error = $state('');
	let warning = $state('');
	let loading = $state(false);

	async function predict() {
		error = warning = '';
		if (!selected.length) {
			warning = 'Veuillez sélectionner au moins un modèle.';
			return;
		}
		if (total <= 0) {
			warning = 'Le budget total doit être supérieur à 0.';
			return;
		}
		loading = true;
		try {
			const r = await apiPost<{ results: Record<string, Result> }>('/sujet-3/predict', {
				tv,
				radio,
				social_media,
				influencer,
				models: selected
			});
			results = r.results;
		} catch (e) {
			results = null;
			error = `Erreur de connexion à l'API : ${(e as Error).message}`;
		} finally {
			loading = false;
		}
	}
</script>

<PageHeader
	icon="trending-up"
	title="Optimisation du ROI Marketing"
	subtitle="Simulez l'impact d'un mix média sur les ventes et estimez le ROI de votre campagne."
/>

<div class="stack">
	<Card title="Budget média" icon="coins">
		<div class="grid cols-4">
			<NumberInput label="Budget TV (M€)" bind:value={tv} min={0} decimals={2} />
			<NumberInput label="Budget Radio (M€)" bind:value={radio} min={0} step={0.5} decimals={2} />
			<NumberInput label="Budget Social Media (M€)" bind:value={social_media} min={0} step={0.1} decimals={2} />
			<SelectInput
				label="Type d'influenceur"
				bind:value={influencer}
				options={['Macro', 'Mega', 'Micro', 'Nano'].map((v) => ({ value: v, label: v }))}
			/>
		</div>
	</Card>

	<Card title="Indicateurs du scénario" icon="bar-chart">
		<div class="grid cols-4">
			<Metric label="Budget total" value="{total.toFixed(1)} M€" />
			<Metric label="Part TV" value={share(tv)} />
			<Metric label="Part Radio" value={share(radio)} />
			<Metric label="Part Social Media" value={share(social_media)} />
		</div>
		{#if total > 0}
			<div class="gap">
				<BarChart
					bars={[
						{ label: 'TV', value: tv },
						{ label: 'Radio', value: radio },
						{ label: 'Social Media', value: social_media }
					]}
					format={(v) => `${v.toFixed(1)} M€`}
				/>
			</div>
		{/if}
	</Card>

	<Card title="Modèles à comparer" icon="layers">
		<MultiSelect label="Modèles de régression" bind:value={selected} options={opts} />
		{#if !available.length}
			<div class="gap">
				<Alert kind="warning">Aucun modèle n'est disponible pour le moment. Veuillez réessayer plus tard.</Alert>
			</div>
		{/if}
		<button class="btn primary block gap" type="button" onclick={predict} disabled={loading}>
			<Icon name="zap" size={18} />
			{loading ? 'Analyse en cours…' : 'Lancer la prédiction'}
		</button>
		{#if warning}<div class="gap"><Alert kind="warning">{warning}</Alert></div>{/if}
		{#if error}<div class="gap"><Alert kind="error">{error}</Alert></div>{/if}
	</Card>

	{#if results}
		<h2>Résultats par modèle</h2>
		<div class="grid auto-fit">
			{#each Object.entries(results) as [key, res] (key)}
				<Card tone={PERF[res.performance]}>
					<div class="stack tight">
						<h3>{lbl(key)}</h3>
						<Metric label="Ventes prédites" value="{res.sales_prediction.toFixed(2)} M€" />
						<Metric label="ROI estimé" value={res.roi_estimate !== null ? `${res.roi_estimate.toFixed(2)}x` : '–'} />
						<Alert kind={PERF[res.performance]}>Performance : <strong>{res.performance}</strong></Alert>
					</div>
				</Card>
			{/each}
		</div>

		{#if Object.keys(results).length > 1}
			<Card title="Tableau comparatif" icon="layers">
				<DataTable
					columns={['Modèle', 'Ventes prédites (M€)', 'ROI estimé', 'Performance']}
					rows={Object.entries(results).map(([k, v]) => [
						lbl(k),
						v.sales_prediction.toFixed(2),
						v.roi_estimate !== null ? v.roi_estimate.toFixed(2) : null,
						v.performance
					])}
				/>
			</Card>
		{/if}
	{/if}
</div>

<style>
	.tight {
		gap: 0.9rem;
	}
</style>
