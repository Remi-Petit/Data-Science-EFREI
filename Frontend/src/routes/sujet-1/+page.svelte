<script lang="ts">
	import { onMount } from 'svelte';
	import { apiGet, apiPost, modelLabel, pct, type ModelEntry } from '$lib/api';
	import machinesData from '$lib/machines.json';
	import Alert from '$lib/components/Alert.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import Metric from '$lib/components/Metric.svelte';
	import MultiSelect from '$lib/components/MultiSelect.svelte';
	import NumberInput from '$lib/components/NumberInput.svelte';
	import Progress from '$lib/components/Progress.svelte';
	import SelectInput from '$lib/components/SelectInput.svelte';
	import Slider from '$lib/components/Slider.svelte';
	import Tabs from '$lib/components/Tabs.svelte';

	type Result = {
		prediction: number;
		label: string;
		probabilite_panne: number;
		rul_heures?: number;
		cause_potentielle?: string;
		probabilites_causes?: Record<string, number>;
	};
	type Machine = (typeof machinesData)[number];
	type MachineResult = Machine & Partial<Result>;

	const machines: Machine[] = machinesData;
	const FALLBACK = ['logistic_regression', 'random_forest', 'xgboost'];

	let tab = $state(0);
	let groups = $state<Record<string, string[]>>({});
	let labels = $state<Record<string, Record<string, string>>>({});
	let causeLabels = $state<Record<string, string>>({});
	const models24h = $derived(groups.failure_24h ?? FALLBACK);
	const modelsCause = $derived(groups.failure_type ?? models24h);
	const modelsRul = $derived(groups.rul ?? models24h);

	onMount(async () => {
		try {
			const { models } = await apiGet<{ models: Record<string, ModelEntry[]> }>('/sujet-1/models');
			const g: Record<string, string[]> = {};
			const l: Record<string, Record<string, string>> = {};
			for (const [name, entries] of Object.entries(models)) {
				g[name] = entries.map((e) => e.name);
				l[name] = Object.fromEntries(entries.map((e) => [e.name, e.label]));
			}
			groups = g;
			labels = l;
			selected = models24h.slice(0, 1);
			dashModel = models24h[0];
			dashCause = modelsCause[0];
			dashRul = modelsRul[0];
		} catch {
			/* valeurs de repli */
		}
		try {
			causeLabels = (await apiGet<{ labels: Record<string, string> }>('/sujet-1/cause-labels')).labels;
		} catch {
			/* libellés bruts */
		}
	});

	const lbl = (key: string, group = 'failure_24h') =>
		modelLabel(key, groups[group] ?? models24h, labels[group] ?? {});
	const opts = (group: string) => (groups[group] ?? models24h).map((k) => ({ value: k, label: lbl(k, group) }));
	const causeName = (k: string) => causeLabels[k] ?? k;
	const causeRows = (scores: Record<string, number>) =>
		Object.entries(scores)
			.sort((a, b) => b[1] - a[1])
			.map(([k, v]) => [causeName(k), `${(v * 100).toFixed(1)}%`]);

	// ── Onglet 1 : prédiction ─────────────────────────────────────────────────
	let vibration_rms = $state(2.0);
	let temperature_motor = $state(70.0);
	let current_phase_avg = $state(10.0);
	let ambient_temp = $state(25.0);
	let pressure_level = $state(3.0);
	let rpm = $state(1450.0);
	let hours_since_maintenance = $state(200.0);
	let machine_type_enc = $state(0);
	let operating_mode_enc = $state(0);
	let hour = $state(12);
	let dayofweek = $state(1);
	let month = $state(6);
	let selected = $state<string[]>(['random_forest']);

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
		loading = true;
		try {
			const r = await apiPost<{ results: Record<string, Result> }>('/sujet-1/predict', {
				vibration_rms,
				temperature_motor,
				current_phase_avg,
				pressure_level,
				rpm,
				hours_since_maintenance,
				ambient_temp,
				machine_type_enc,
				operating_mode_enc,
				hour,
				dayofweek,
				month,
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

	// ── Onglet 2 : dashboard ──────────────────────────────────────────────────
	let dashModel = $state('random_forest');
	let dashCause = $state('random_forest');
	let dashRul = $state('random_forest');
	let dashResults = $state<MachineResult[] | null>(null);
	let progress = $state<{ done: number; total: number } | null>(null);

	async function analyzeAll() {
		const out: MachineResult[] = [];
		progress = { done: 0, total: machines.length };
		for (const m of machines) {
			try {
				const r = await apiPost<{ results: Record<string, Result> }>('/sujet-1/predict', {
					vibration_rms: m.vibration_rms,
					temperature_motor: m.temperature_motor,
					current_phase_avg: m.current_phase_avg,
					pressure_level: m.pressure_level,
					rpm: m.rpm,
					hours_since_maintenance: m.hours_since_maintenance,
					ambient_temp: m.ambient_temp,
					machine_type_enc: m.type,
					operating_mode_enc: m.operating_mode_enc,
					hour: m.hour,
					dayofweek: m.dayofweek,
					month: m.month,
					models: [dashModel],
					model_cause: dashCause,
					model_rul: dashRul
				});
				out.push({ ...m, ...r.results[dashModel] });
			} catch (e) {
				out.push({ ...m, prediction: -1, probabilite_panne: 0, label: `Erreur: ${(e as Error).message}` });
			}
			progress = { done: out.length, total: machines.length };
		}
		progress = null;
		dashResults = out;
	}

	const nOk = $derived(dashResults?.filter((r) => r.prediction === 0).length ?? 0);
	const nRisk = $derived(dashResults?.filter((r) => r.prediction === 1).length ?? 0);
	const avgProb = $derived(
		dashResults?.length ? dashResults.reduce((s, r) => s + (r.probabilite_panne ?? 0), 0) / dashResults.length : 0
	);
</script>

<h1>🔧 Prédiction de panne machine</h1>
<hr />

<Tabs items={['🔍 Prédiction', '📟 Dashboard machines']} bind:active={tab} />

{#if tab === 0}
	<p>Renseignez les paramètres de la machine pour prédire un risque de panne dans les 24h.</p>
	<hr />

	<div class="grid cols-3">
		<div>
			<NumberInput label="Vibration RMS" bind:value={vibration_rms} step={0.1} decimals={2} />
			<NumberInput label="Température moteur (°C)" bind:value={temperature_motor} step={0.5} decimals={2} />
			<NumberInput label="Courant moyen (A)" bind:value={current_phase_avg} step={0.1} decimals={2} />
			<NumberInput label="Température ambiante (°C)" bind:value={ambient_temp} step={0.5} decimals={2} />
		</div>
		<div>
			<NumberInput label="Pression (bar)" bind:value={pressure_level} step={0.1} decimals={2} />
			<NumberInput label="RPM" bind:value={rpm} step={10} decimals={2} />
			<NumberInput label="Heures depuis maintenance" bind:value={hours_since_maintenance} step={10} decimals={2} />
		</div>
		<div>
			<SelectInput
				label="Type de machine"
				bind:value={machine_type_enc}
				options={[
					{ value: 0, label: 'Pump' },
					{ value: 1, label: 'Compressor' },
					{ value: 2, label: 'Motor' }
				]}
			/>
			<SelectInput
				label="Mode opératoire"
				bind:value={operating_mode_enc}
				options={[
					{ value: 0, label: 'normal' },
					{ value: 1, label: 'idle' },
					{ value: 2, label: 'peak' }
				]}
			/>
			<Slider label="Heure" bind:value={hour} min={0} max={23} />
			<Slider label="Jour de la semaine" bind:value={dayofweek} min={0} max={6} />
			<Slider label="Mois" bind:value={month} min={1} max={12} />
		</div>
	</div>

	<hr />
	<MultiSelect label="🤖 Modèles à comparer" bind:value={selected} options={opts('failure_24h')} />
	<hr />

	<button class="btn block" type="button" onclick={predict} disabled={loading}>
		{loading ? 'Analyse…' : '🔍 Lancer la prédiction'}
	</button>

	{#if warning}<div class="gap"><Alert kind="warning">{warning}</Alert></div>{/if}
	{#if error}<div class="gap"><Alert kind="error">{error}</Alert></div>{/if}

	{#if results}
		<hr />
		<h2>📊 Résultats par modèle</h2>
		<div class="grid" style="grid-template-columns: repeat({Object.keys(results).length}, minmax(0, 1fr))">
			{#each Object.entries(results) as [key, res] (key)}
				<div>
					<h3>{lbl(key)}</h3>
					<Alert kind={res.prediction === 1 ? 'error' : 'success'}>
						{res.prediction === 1 ? '⚠️' : '✅'} {res.label}
					</Alert>
					<Metric label="Probabilité de panne" value={pct(res.probabilite_panne)} />
					{#if res.rul_heures !== undefined}
						<Metric label="⏱️ Durée de vie restante" value="{res.rul_heures.toFixed(1)} h" />
					{/if}
					{#if res.prediction === 1 && res.cause_potentielle}
						<p><strong>Cause potentielle la plus probable :</strong></p>
						<Alert kind="info">🔎 {causeName(res.cause_potentielle)}</Alert>
						{#if res.probabilites_causes}
							<p><strong>Détail des probabilités (toutes causes) :</strong></p>
							<DataTable columns={['Cause', 'Probabilité (%)']} rows={causeRows(res.probabilites_causes)} />
						{/if}
					{/if}
				</div>
			{/each}
		</div>
	{/if}
{:else}
	<p>État en temps réel des 10 machines de l'usine.</p>
	<hr />

	<div class="grid cols-3">
		<SelectInput label="🤖 Détection de panne" bind:value={dashModel} options={opts('failure_24h')} />
		<SelectInput label="🔎 Cause de panne" bind:value={dashCause} options={opts('failure_type')} />
		<SelectInput label="⏱️ Durée de vie restante" bind:value={dashRul} options={opts('rul')} />
	</div>

	<button class="btn block" type="button" onclick={analyzeAll} disabled={progress !== null}>
		🔄 Analyser toutes les machines
	</button>

	{#if progress}
		<div class="gap">
			<Progress value={progress.done / progress.total} text="Analyse en cours… {progress.done}/{progress.total}" />
		</div>
	{/if}

	{#if dashResults}
		<div class="grid cols-4 gap">
			<Metric label="Machines analysées" value={dashResults.length} />
			<Metric label="✅ En bon état" value={nOk} />
			<Metric label="⚠️ À risque" value={nRisk} />
			<Metric label="Prob. panne moy." value="{(avgProb * 100).toFixed(1)}%" />
		</div>
		<hr />
		<div class="grid cols-2">
			{#each dashResults as r (r.id)}
				{@const pred = r.prediction ?? -1}
				<div class="card">
					<div class="card-head">
						<div>
							<strong>{pred === 0 ? '✅' : pred === 1 ? '⚠️' : '❓'} {r.id} – {r.name}</strong>
							<div class="caption">{r.type_label} · {r.location}</div>
						</div>
						<div class="badge {pred === 0 ? 'success' : pred === 1 ? 'error' : 'warning'}">
							{((r.probabilite_panne ?? 0) * 100).toFixed(0)}%
						</div>
					</div>
					<div class="grid cols-3">
						<Metric label="Vibration" value="{r.vibration_rms} g" />
						<Metric label="Température" value="{r.temperature_motor} °C" />
						<Metric label="Maintenance" value="{r.hours_since_maintenance} h" />
					</div>
					{#if r.rul_heures !== undefined}
						<Metric label="⏱️ Durée de vie restante" value="{r.rul_heures.toFixed(1)} h" />
					{/if}
					{#if pred === 1 && r.cause_potentielle}
						<Alert kind="info">🔎 Cause probable : {causeName(r.cause_potentielle)}</Alert>
						{#if r.probabilites_causes}
							<details>
								<summary>Détail des probabilités par cause</summary>
								<DataTable columns={['Cause', 'Probabilité']} rows={causeRows(r.probabilites_causes)} />
							</details>
						{/if}
					{/if}
				</div>
			{/each}
		</div>
	{:else}
		<div class="gap"><Alert kind="info">Cliquez sur <strong>Analyser toutes les machines</strong> pour lancer les prédictions.</Alert></div>
		<DataTable
			columns={['ID', 'Nom', 'Type', 'Localisation', 'Vibration (g)', 'Temp. moteur (°C)', 'H. depuis maint.']}
			rows={machines.map((m) => [m.id, m.name, m.type_label, m.location, m.vibration_rms, m.temperature_motor, m.hours_since_maintenance])}
		/>
	{/if}
{/if}

<style>
	.gap {
		margin-top: 1rem;
	}
	.card {
		padding: 1rem;
		border: 1px solid var(--border);
		border-radius: var(--radius);
	}
	.card-head {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 1rem;
		margin-bottom: 0.5rem;
	}
	.badge {
		padding: 0.25rem 0.75rem;
		border-radius: var(--radius);
		font-weight: 600;
	}
	.badge.success {
		background: var(--success-bg);
		color: var(--success-fg);
	}
	.badge.error {
		background: var(--error-bg);
		color: var(--error-fg);
	}
	.badge.warning {
		background: var(--warning-bg);
		color: var(--warning-fg);
	}
	details {
		margin-top: 0.5rem;
	}
	summary {
		cursor: pointer;
		margin-bottom: 0.5rem;
	}
	@media (max-width: 600px) {
		.grid[style] {
			grid-template-columns: minmax(0, 1fr) !important;
		}
	}
</style>
