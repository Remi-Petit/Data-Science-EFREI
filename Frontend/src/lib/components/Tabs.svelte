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
		gap: 1rem;
		overflow-x: auto;
		scrollbar-width: none;
		border-bottom: 1px solid var(--border);
		margin-bottom: 1.5rem;
	}
	button {
		padding: 0.5rem 0.25rem;
		margin-bottom: -1px;
		border: none;
		border-bottom: 2px solid transparent;
		background: transparent;
		color: var(--text-muted);
		cursor: pointer;
		white-space: nowrap;
	}
	button:hover {
		color: var(--primary);
	}
	button.active {
		color: var(--primary);
		border-bottom-color: var(--primary);
	}
</style>
