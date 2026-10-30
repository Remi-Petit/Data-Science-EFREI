<script lang="ts">
	let { items, active = $bindable() }: { items: string[]; active: number } = $props();
</script>

<div class="tabs" role="tablist">
	{#each items as item, i (item)}
		<button
			type="button"
			role="tab"
			aria-selected={active === i}
			class:active={active === i}
			onclick={() => (active = i)}>{item}</button
		>
	{/each}
</div>

<style>
	.tabs {
		display: flex;
		width: fit-content;
		max-width: 100%;
		gap: 0.25rem;
		padding: 0.3rem;
		margin-bottom: 1.5rem;
		overflow-x: auto;
		scrollbar-width: none;
		background: var(--bg-secondary);
		border-radius: var(--radius);
	}
	button {
		padding: 0.45rem 1rem;
		border: none;
		border-radius: 0.55rem;
		background: transparent;
		color: var(--text-muted);
		font-weight: 600;
		white-space: nowrap;
		cursor: pointer;
		transition: background 0.15s, color 0.15s, box-shadow 0.15s;
	}
	button:hover {
		color: var(--text);
	}
	button.active {
		background: var(--surface);
		color: var(--primary);
		box-shadow: var(--shadow);
	}
</style>
