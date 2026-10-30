<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { apiGet } from '$lib/api';
	import { theme } from '$lib/theme.svelte';
	import Icon from '$lib/components/Icon.svelte';

	let { children } = $props();

	const pages = [
		{ href: '/sujet-1', icon: 'wrench', title: 'Maintenance prédictive', tag: 'Sujet 1' },
		{ href: '/sujet-2', icon: 'trending-down', title: 'Churn client', tag: 'Sujet 2' },
		{ href: '/sujet-3', icon: 'trending-up', title: 'ROI Marketing', tag: 'Sujet 3' },
		{ href: '/modeles', icon: 'bar-chart', title: 'Comparaison des modèles', tag: 'Analyse' }
	];

	let status = $state<'loading' | 'ok' | 'down'>('loading');
	let menuOpen = $state(false);

	onMount(async () => {
		theme.init();
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
	<button class="burger" type="button" aria-label="Menu" onclick={() => (menuOpen = !menuOpen)}>
		<Icon name={menuOpen ? 'x' : 'menu'} size={20} />
	</button>
	{#if menuOpen}
		<button class="backdrop" type="button" aria-label="Fermer le menu" onclick={() => (menuOpen = false)}></button>
	{/if}

	<aside class:open={menuOpen}>
		<a class="brand" href="/sujet-1">
			<span class="logo"><Icon name="cpu" size={22} /></span>
			<span>
				<strong>Data Science</strong>
				<small>EFREI · M1 DEV</small>
			</span>
		</a>

		<nav>
			<span class="section">Projets</span>
			{#each pages as p (p.href)}
				<a href={p.href} class:active={page.url.pathname.startsWith(p.href)}>
					<span class="icon"><Icon name={p.icon} size={18} /></span>
					<span class="text">
						<small>{p.tag}</small>
						{p.title}
					</span>
				</a>
			{/each}
		</nav>

		<div class="bottom">
			<div class="status {status}">
				<span class="dot"></span>
				{status === 'ok' ? 'API opérationnelle' : status === 'down' ? 'API inaccessible' : 'Connexion à l’API…'}
			</div>

			<button class="theme" type="button" onclick={() => theme.toggle()} aria-label="Changer de thème">
				<span class="label">
					<Icon name={theme.current === 'dark' ? 'moon' : 'sun'} size={16} />
					{theme.current === 'dark' ? 'Mode sombre' : 'Mode clair'}
				</span>
				<span class="switch" class:on={theme.current === 'dark'}><span class="knob"></span></span>
			</button>

			<div class="links">
				<a class="btn" href="https://remipetit.fr/data-science-ia/" target="_blank" rel="noreferrer">
					<Icon name="globe" size={16} /> Portfolio
				</a>
				<a class="btn" href="https://github.com/Remi-Petit/Data-Science-EFREI" target="_blank" rel="noreferrer">
					<Icon name="github" size={16} /> GitHub
				</a>
			</div>
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
		gap: 1.5rem;
		flex: 0 0 18.5rem;
		height: 100vh;
		overflow-y: auto;
		padding: 1.5rem 1rem 1rem;
		background: var(--surface);
		border-right: 1px solid var(--border);
		transition: background-color 0.25s, border-color 0.25s;
	}

	.brand {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0 0.5rem;
		color: var(--text);
		text-decoration: none;
	}
	.logo {
		display: grid;
		place-items: center;
		width: 2.6rem;
		height: 2.6rem;
		border-radius: 0.8rem;
		background: var(--gradient);
		color: #fff;
		box-shadow: 0 8px 18px -8px var(--primary);
	}
	.brand strong {
		display: block;
		font-size: 1.05rem;
		letter-spacing: -0.02em;
		line-height: 1.2;
	}
	.brand small {
		color: var(--text-muted);
		font-size: 0.75rem;
	}

	nav {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
	}
	.section {
		padding: 0 0.75rem 0.4rem;
		font-size: 0.7rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--text-muted);
	}
	nav a {
		position: relative;
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.6rem 0.75rem;
		border-radius: var(--radius);
		color: var(--text);
		text-decoration: none;
		font-weight: 550;
		transition: background 0.15s, color 0.15s;
	}
	nav a .icon {
		display: grid;
		place-items: center;
		flex: none;
		width: 2.1rem;
		height: 2.1rem;
		border-radius: 0.6rem;
		background: var(--bg-secondary);
		color: var(--text-muted);
		transition: background 0.15s, color 0.15s;
	}
	nav a .text {
		line-height: 1.25;
	}
	nav a small {
		display: block;
		font-size: 0.68rem;
		font-weight: 600;
		letter-spacing: 0.05em;
		text-transform: uppercase;
		color: var(--text-muted);
	}
	nav a:hover {
		background: var(--bg-secondary);
	}
	nav a.active {
		background: var(--primary-soft);
		color: var(--primary);
	}
	nav a.active .icon {
		background: var(--gradient);
		color: #fff;
	}
	nav a.active small {
		color: var(--primary);
	}

	.bottom {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		margin-top: auto;
	}

	.status {
		display: flex;
		align-items: center;
		gap: 0.55rem;
		padding: 0.55rem 0.8rem;
		border-radius: var(--radius);
		background: var(--bg-secondary);
		font-size: 0.85rem;
		font-weight: 550;
	}
	.dot {
		width: 0.55rem;
		height: 0.55rem;
		border-radius: 50%;
		background: var(--text-muted);
	}
	.status.ok .dot {
		background: var(--success);
		box-shadow: 0 0 0 3px var(--success-bg);
	}
	.status.down .dot {
		background: var(--error);
		box-shadow: 0 0 0 3px var(--error-bg);
	}

	.theme {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.55rem 0.8rem;
		border: 1px solid var(--border);
		border-radius: var(--radius);
		background: transparent;
		cursor: pointer;
		font-size: 0.85rem;
		font-weight: 550;
		transition: border-color 0.15s, background 0.15s;
	}
	.theme:hover {
		border-color: var(--primary);
		background: var(--primary-soft);
	}
	.theme .label {
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
	}
	.switch {
		position: relative;
		width: 2.5rem;
		height: 1.4rem;
		border-radius: 999px;
		background: var(--bg-secondary);
		border: 1px solid var(--border);
		transition: background 0.2s;
	}
	.switch.on {
		background: var(--primary);
		border-color: var(--primary);
	}
	.knob {
		position: absolute;
		top: 0.12rem;
		left: 0.15rem;
		width: 1rem;
		height: 1rem;
		border-radius: 50%;
		background: #fff;
		box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
		transition: transform 0.2s;
	}
	.switch.on .knob {
		transform: translateX(1.1rem);
	}

	.links {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.5rem;
	}
	.links .btn {
		min-height: 2.3rem;
		padding: 0.35rem 0.5rem;
		font-size: 0.85rem;
	}

	main {
		flex: 1;
		min-width: 0;
	}
	.content {
		max-width: 1180px;
		margin: 0 auto;
		padding: 2.5rem 3rem 5rem;
	}

	.burger {
		display: none;
	}
	.backdrop {
		display: none;
	}

	@media (max-width: 900px) {
		.content {
			padding: 4.5rem 1.1rem 4rem;
		}
		.burger {
			display: grid;
			place-items: center;
			position: fixed;
			top: 0.75rem;
			left: 0.75rem;
			z-index: 40;
			width: 2.6rem;
			height: 2.6rem;
			border: 1px solid var(--border);
			border-radius: var(--radius);
			background: var(--surface);
			box-shadow: var(--shadow);
			cursor: pointer;
		}
		.backdrop {
			display: block;
			position: fixed;
			inset: 0;
			z-index: 25;
			border: none;
			background: rgba(2, 6, 23, 0.55);
		}
		aside {
			position: fixed;
			z-index: 30;
			left: 0;
			top: 0;
			transform: translateX(-100%);
			transition: transform 0.25s ease;
			width: min(19rem, 85vw);
			flex-basis: auto;
			padding-top: 4.5rem;
			box-shadow: var(--shadow-lg);
		}
		aside.open {
			transform: none;
		}
	}
</style>
