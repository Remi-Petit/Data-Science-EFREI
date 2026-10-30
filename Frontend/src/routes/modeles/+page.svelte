<script lang="ts">
	import { onMount } from 'svelte';
	import { apiGet, type ModelEntry, type Stats } from '$lib/api';
	import Alert from '$lib/components/Alert.svelte';
	import StatsGroup from '$lib/components/StatsGroup.svelte';
	import Tabs from '$lib/components/Tabs.svelte';

	type Labels = Record<string, string>;
	type Subject<S> = { labels: Labels; groupLabels: Record<string, Labels>; stats: S } | null;

	let tab = $state(0);
	let sub1 = $state(0);
	let sub3 = $state(0);
	let loaded = $state(false);

	let s1 = $state<Subject<Record<string, Stats>>>(null);
	let s2 = $state<Subject<Stats>>(null);
	let s3 = $state<Subject<Record<string, Stats>>>(null);

	async function fetchSubject<S>(n: number): Promise<Subject<S>> {
		try {
			const [m, s] = await Promise.all([
				apiGet<{ models: ModelEntry[] | Record<string, ModelEntry[]> }>(`/sujet-${n}/models`),
				apiGet<{ stats: S }>(`/sujet-${n}/stats`)
			]);
			const toMap = (entries: ModelEntry[]) => Object.fromEntries(entries.map((e) => [e.name, e.label]));
			const groups = Array.isArray(m.models) ? {} : m.models;
			return {
				labels: toMap(Array.isArray(m.models) ? m.models : []),
				groupLabels: Object.fromEntries(Object.entries(groups).map(([g, e]) => [g, toMap(e)])),
				stats: s.stats
			};
		} catch {
			return null;
		}
	}

	onMount(async () => {
		[s1, s2, s3] = await Promise.all([
			fetchSubject<Record<string, Stats>>(1),
			fetchSubject<Stats>(2),
			fetchSubject<Record<string, Stats>>(3)
		]);
		loaded = true;
	});
</script>

<h1>📊 Comparaison des modèles</h1>
<p>Classement des modèles entraînés pour chaque sujet, du meilleur au moins performant.</p>
<hr />

<Tabs items={['🔧 Sujet 1 – Maintenance', '📉 Sujet 2 – Churn', '📈 Sujet 3 – Marketing']} bind:active={tab} />

{#if !loaded}
	<p class="caption">Chargement…</p>
{:else if tab === 0}
	{#if s1}
		<Tabs items={['Détection panne 24h', 'Type de panne', 'Durée de vie (RUL)']} bind:active={sub1} />
		{#if sub1 === 0}
			<StatsGroup group={s1.stats.failure_24h ?? {}} labels={s1.groupLabels.failure_24h ?? {}} rankCol="F1-score" />
		{:else if sub1 === 1}
			<StatsGroup group={s1.stats.failure_type ?? {}} labels={s1.groupLabels.failure_type ?? {}} rankCol="F1-score" />
		{:else}
			<StatsGroup group={s1.stats.rul ?? {}} labels={s1.groupLabels.rul ?? {}} rankCol="R²" asPct={false} />
		{/if}
	{:else}
		<Alert kind="warning">⚠️ Les statistiques du Sujet 1 ne sont pas disponibles pour le moment.</Alert>
	{/if}
{:else if tab === 1}
	{#if s2}
		<StatsGroup group={s2.stats} labels={s2.labels} rankCol="F1-score" />
	{:else}
		<Alert kind="warning">⚠️ Les statistiques du Sujet 2 ne sont pas disponibles pour le moment.</Alert>
	{/if}
{:else if s3}
	<Tabs items={['Régression (Ventes)', 'Classification (Performance campagne)']} bind:active={sub3} />
	{#if sub3 === 0}
		<StatsGroup group={s3.stats.regression ?? {}} labels={s3.labels} rankCol="R²" asPct={false} />
	{:else}
		<StatsGroup group={s3.stats.classification ?? {}} labels={s3.labels} rankCol="F1 (macro)" />
	{/if}
{:else}
	<Alert kind="warning">⚠️ Les statistiques du Sujet 3 ne sont pas disponibles pour le moment.</Alert>
{/if}
