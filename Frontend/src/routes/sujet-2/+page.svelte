<script lang="ts">
	import { onMount } from 'svelte';
	import { apiGet, apiPost, modelLabel, type ModelEntry, type Stats } from '$lib/api';
	import Alert from '$lib/components/Alert.svelte';
	import Card from '$lib/components/Card.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import MultiSelect from '$lib/components/MultiSelect.svelte';
	import NumberInput from '$lib/components/NumberInput.svelte';
	import PageHeader from '$lib/components/PageHeader.svelte';
	import ProbBar from '$lib/components/ProbBar.svelte';
	import SelectInput from '$lib/components/SelectInput.svelte';

	type Result = { prediction: number; label: string; probabilite_churn: number };

	const FALLBACK = ['logistic_regression', 'random_forest', 'xgboost', 'mlp'];
	let available = $state<string[]>(FALLBACK);
	let labels = $state<Record<string, string>>({});
	let selected = $state<string[]>(['random_forest']);

	onMount(async () => {
		try {
			const { models } = await apiGet<{ models: ModelEntry[] }>('/sujet-2/models');
			const names = models.map((e) => e.name);
			labels = Object.fromEntries(models.map((e) => [e.name, e.label]));
			try {
				const { stats } = await apiGet<{ stats: Stats }>('/sujet-2/stats');
				names.sort((a, b) => (stats[b]?.['F1-score'] ?? 0) - (stats[a]?.['F1-score'] ?? 0));
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
	const yesNo = [
		{ value: 'No', label: 'No' },
		{ value: 'Yes', label: 'Yes' }
	];
	const simple = (...v: string[]) => v.map((x) => ({ value: x, label: x }));

	let f = $state({
		age: 35,
		tenure_months: 24,
		monthly_fee: 49.0,
		total_revenue: 1200.0,
		gender: 'Female',
		customer_segment: 'Enterprise',
		monthly_logins: 15,
		weekly_active_days: 3,
		avg_session_time: 20.0,
		features_used: 5,
		usage_growth_rate: 0.05,
		last_login_days_ago: 5,
		payment_failures: 0,
		support_tickets: 1,
		avg_resolution_time: 24.0,
		csat_score: 4.0,
		escalations: 0,
		nps_score: 7,
		email_open_rate: 0.3,
		marketing_click_rate: 0.1,
		referral_count: 0,
		signup_channel: 'Mobile',
		contract_type: 'Monthly',
		payment_method: 'Bank Transfer',
		discount_applied: 'No',
		price_increase_last_3m: 'No',
		survey_response: 'Neutral',
		complaint_type: 'Billing'
	});

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
			const r = await apiPost<{ results: Record<string, Result> }>('/sujet-2/predict', { ...f, models: selected });
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
	icon="trending-down"
	title="Churn client"
	subtitle="Renseignez le profil du client pour prédire s'il risque de résilier son abonnement."
/>

<div class="stack">
	<div class="grid cols-3">
		<Card title="Profil client" icon="user">
			<div class="stack tight">
				<NumberInput label="Âge" bind:value={f.age} min={18} max={100} />
				<NumberInput label="Ancienneté (mois)" bind:value={f.tenure_months} min={0} />
				<NumberInput label="Frais mensuels (€)" bind:value={f.monthly_fee} decimals={2} />
				<NumberInput label="Revenu total généré (€)" bind:value={f.total_revenue} step={50} decimals={2} />
				<SelectInput label="Genre" bind:value={f.gender} options={simple('Female', 'Male')} />
				<SelectInput label="Segment client" bind:value={f.customer_segment} options={simple('Enterprise', 'Individual', 'SME')} />
			</div>
		</Card>

		<Card title="Usage" icon="trending-up">
			<div class="stack tight">
				<NumberInput label="Connexions / mois" bind:value={f.monthly_logins} min={0} />
				<NumberInput label="Jours actifs / semaine" bind:value={f.weekly_active_days} min={0} max={7} />
				<NumberInput label="Durée moy. session (min)" bind:value={f.avg_session_time} decimals={2} />
				<NumberInput label="Fonctionnalités utilisées" bind:value={f.features_used} min={0} />
				<NumberInput label="Taux de croissance d'usage (%)" bind:value={f.usage_growth_rate} step={0.01} decimals={2} />
				<NumberInput label="Dernière connexion (jours)" bind:value={f.last_login_days_ago} min={0} />
			</div>
		</Card>

		<Card title="Support & satisfaction" icon="info">
			<div class="stack tight">
				<NumberInput label="Échecs de paiement" bind:value={f.payment_failures} min={0} />
				<NumberInput label="Tickets support" bind:value={f.support_tickets} min={0} />
				<NumberInput label="Temps résolution moy. (h)" bind:value={f.avg_resolution_time} decimals={2} />
				<NumberInput label="Score CSAT" bind:value={f.csat_score} step={0.1} min={1} max={5} decimals={2} />
				<NumberInput label="Escalades" bind:value={f.escalations} min={0} />
				<NumberInput label="NPS Score" bind:value={f.nps_score} min={0} max={10} />
			</div>
		</Card>
	</div>

	<Card title="Engagement & marketing" icon="zap">
		<div class="grid cols-3">
			<div class="stack tight">
				<NumberInput label="Taux d'ouverture email" bind:value={f.email_open_rate} step={0.01} min={0} max={1} decimals={2} />
				<NumberInput label="Taux de clic marketing" bind:value={f.marketing_click_rate} step={0.01} min={0} max={1} decimals={2} />
				<NumberInput label="Parrainages" bind:value={f.referral_count} min={0} />
			</div>
			<div class="stack tight">
				<SelectInput label="Canal d'acquisition" bind:value={f.signup_channel} options={simple('Mobile', 'Referral', 'Web')} />
				<SelectInput label="Type de contrat" bind:value={f.contract_type} options={simple('Monthly', 'Quarterly', 'Yearly')} />
				<SelectInput label="Méthode de paiement" bind:value={f.payment_method} options={simple('Bank Transfer', 'Card', 'PayPal')} />
			</div>
			<div class="stack tight">
				<SelectInput label="Remise appliquée" bind:value={f.discount_applied} options={yesNo} />
				<SelectInput label="Hausse de prix (3 derniers mois)" bind:value={f.price_increase_last_3m} options={yesNo} />
				<SelectInput label="Réponse enquête" bind:value={f.survey_response} options={simple('Neutral', 'Satisfied', 'Unsatisfied')} />
				<SelectInput label="Type de réclamation" bind:value={f.complaint_type} options={simple('Billing', 'Service', 'Technical', 'Unknown')} />
			</div>
		</div>
	</Card>

	<Card title="Modèles à comparer" icon="layers">
		<MultiSelect label="Modèles de prédiction" bind:value={selected} options={opts} />
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
				<Card tone={res.prediction === 1 ? 'error' : 'success'}>
					<div class="stack tight">
						<h3>{lbl(key)}</h3>
						<Alert kind={res.prediction === 1 ? 'error' : 'success'}>{res.label}</Alert>
						<ProbBar label="Probabilité de churn" value={res.probabilite_churn} />
					</div>
				</Card>
			{/each}
		</div>
	{/if}
</div>

<style>
	.tight {
		gap: 0.9rem;
	}
</style>
