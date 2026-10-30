<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { apiGet } from '$lib/api';
	import Alert from '$lib/components/Alert.svelte';

	let { children } = $props();

	const pages = [
		{ href: '/sujet-1', icon: '🔧', title: 'Sujet 1 – Maintenance prédictive' },
		{ href: '/sujet-2', icon: '📉', title: 'Sujet 2 – Churn client' },
		{ href: '/sujet-3', icon: '📈', title: 'Sujet 3 – ROI Marketing' },
		{ href: '/modeles', icon: '📊', title: 'Comparaison des modèles' }
	];

	let status = $state<'loading' | 'ok' | 'down'>('loading');
	let menuOpen = $state(false);

	onMount(async () => {
		try {
			await apiGet('/health');
			status = 'ok';
		} catch {
			status = 'down';
		}
	});

	$effect(() => {
		page.url.pathname;
		menuOpen = false;
	});
</script>

<div class="shell">
	<button class="burger" type="button" aria-label="Menu" onclick={() => (menuOpen = !menuOpen)}>☰</button>

	<aside class:open={menuOpen}>
		<nav>
			{#each pages as p (p.href)}
				<a href={p.href} class:active={page.url.pathname.startsWith(p.href)}>
					<span class="icon">{p.icon}</span>{p.title}
				</a>
			{/each}
		</nav>

		<div class="bottom">
			{#if status === 'ok'}
				<Alert kind="success">✅ API opérationnelle</Alert>
			{:else if status === 'down'}
				<Alert kind="error">❌ API inaccessible</Alert>
			{/if}
			<a class="btn block" href="https://remipetit.fr/data-science-ia/" target="_blank" rel="noreferrer">🌐 Portfolio</a>
			<a class="btn block" href="https://github.com/Remi-Petit/Data-Science-EFREI" target="_blank" rel="noreferrer"
				>🐙 GitHub</a
			>
		</div>
	</aside>

	<main>
		<div class="content">{@render children()}</div>
	</main>
</div>

<style>
	.shell {
		display: flex;
		min-height: 100vh;
	}
	aside {
		position: sticky;
		top: 0;
		align-self: flex-start;
		display: flex;
		flex-direction: column;
		gap: 1rem;
		flex: 0 0 21rem;
		height: 100vh;
		overflow-y: auto;
		padding: 2rem 1rem 1rem;
		background: var(--bg-secondary);
	}
	nav {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}
	nav a {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.75rem;
		border-radius: var(--radius);
		color: var(--text);
		text-decoration: none;
	}
	nav a:hover {
		background: rgba(151, 166, 195, 0.15);
	}
	nav a.active {
		background: rgba(151, 166, 195, 0.25);
		font-weight: 600;
	}
	.icon {
		font-size: 1.1rem;
	}
	.bottom {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		margin-top: 1rem;
	}
	.bottom :global(.alert) {
		margin-bottom: 0;
	}
	main {
		flex: 1;
		min-width: 0;
	}
	.content {
		max-width: 1200px;
		margin: 0 auto;
		padding: 3rem 5rem 6rem;
	}
	.burger {
		display: none;
	}

	@media (max-width: 900px) {
		.content {
			padding: 4rem 1.25rem 4rem;
		}
		.burger {
			display: block;
			position: fixed;
			top: 0.5rem;
			left: 0.5rem;
			z-index: 30;
			padding: 0.25rem 0.6rem;
			font-size: 1.25rem;
			background: var(--bg);
			border: 1px solid var(--border);
			border-radius: var(--radius);
			cursor: pointer;
		}
		aside {
			position: fixed;
			z-index: 25;
			left: 0;
			transform: translateX(-100%);
			transition: transform 0.2s;
			width: min(21rem, 85vw);
			flex-basis: auto;
			box-shadow: 0 0 24px rgba(0, 0, 0, 0.2);
		}
		aside.open {
			transform: none;
		}
	}
</style>
