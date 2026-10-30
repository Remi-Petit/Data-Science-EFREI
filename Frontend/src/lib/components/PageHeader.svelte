<script lang="ts">
	import Icon from './Icon.svelte';

	let { icon, title, subtitle }: { icon: string; title: string; subtitle?: string } = $props();
</script>

<header>
	<span class="badge"><Icon name={icon} size={26} /></span>
	<div>
		<h1>{title}</h1>
		{#if subtitle}<p>{subtitle}</p>{/if}
	</div>
</header>

<style>
	header {
		display: flex;
		align-items: center;
		gap: 1.1rem;
		margin-bottom: 2rem;
	}
	.badge {
		display: grid;
		place-items: center;
		flex: none;
		width: 3.5rem;
		height: 3.5rem;
		border-radius: 1rem;
		background: var(--gradient);
		color: #fff;
		box-shadow: 0 10px 24px -10px var(--primary);
	}
	h1 {
		font-size: 1.9rem;
		font-weight: 750;
		letter-spacing: -0.025em;
	}
	p {
		margin: 0.2rem 0 0;
		color: var(--text-muted);
	}
	@media (max-width: 640px) {
		h1 {
			font-size: 1.4rem;
		}
	}
</style>
