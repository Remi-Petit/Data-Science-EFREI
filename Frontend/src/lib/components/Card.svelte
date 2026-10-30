<script lang="ts">
	import type { Snippet } from 'svelte';
	import Icon from './Icon.svelte';

	let {
		title,
		icon,
		tone,
		children
	}: { title?: string; icon?: string; tone?: 'success' | 'error' | 'warning' | 'info'; children: Snippet } = $props();
</script>

<section class="card" class:tone style={tone ? `--accent: var(--${tone})` : ''}>
	{#if title}
		<header>
			{#if icon}<span class="chip"><Icon name={icon} size={16} /></span>{/if}
			<h3>{title}</h3>
		</header>
	{/if}
	{@render children()}
</section>

<style>
	.card {
		position: relative;
		padding: 1.5rem;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		box-shadow: var(--shadow);
		transition: background-color 0.25s, border-color 0.25s;
	}
	.card.tone {
		border-top: 3px solid var(--accent);
	}
	header {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		margin-bottom: 1.25rem;
	}
	.chip {
		display: inline-grid;
		place-items: center;
		width: 1.9rem;
		height: 1.9rem;
		border-radius: 0.55rem;
		background: var(--primary-soft);
		color: var(--primary);
	}
</style>
