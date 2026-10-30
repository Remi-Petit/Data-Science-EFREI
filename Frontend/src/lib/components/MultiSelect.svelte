<script lang="ts">
	type Option = { value: string; label: string };

	let { label, value = $bindable(), options }: { label: string; value: string[]; options: Option[] } = $props();

	let open = $state(false);
	let root: HTMLDivElement;

	const labelOf = (v: string) => options.find((o) => o.value === v)?.label ?? v;
	const remaining = $derived(options.filter((o) => !value.includes(o.value)));

	function add(v: string) {
		value = [...value, v];
	}
	function remove(v: string) {
		value = value.filter((x) => x !== v);
	}
	function onwindowclick(e: MouseEvent) {
		if (open && root && !root.contains(e.target as Node)) open = false;
	}
</script>

<svelte:window onclick={onwindowclick} />

<div class="field" bind:this={root}>
	<span class="label">{label}</span>
	<div
		class="box"
		role="combobox"
		aria-expanded={open}
		aria-controls="ms-list"
		tabindex="0"
		onclick={() => (open = !open)}
		onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && ((open = !open), e.preventDefault())}
	>
		<div class="tags">
			{#each value as v (v)}
				<span class="tag">
					{labelOf(v)}
					<button
						type="button"
						aria-label="Retirer {labelOf(v)}"
						onclick={(e) => {
							e.stopPropagation();
							remove(v);
						}}>×</button
					>
				</span>
			{/each}
		</div>
		<span class="chevron">⌄</span>
	</div>
	{#if open}
		<ul id="ms-list" class="list" role="listbox">
			{#each remaining as o (o.value)}
				<li>
					<button
						type="button"
						role="option"
						aria-selected="false"
						onclick={() => {
							add(o.value);
							if (remaining.length === 1) open = false;
						}}>{o.label}</button
					>
				</li>
			{:else}
				<li class="empty">Aucune option restante</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.field {
		position: relative;
		margin-bottom: 1rem;
	}
	.label {
		display: block;
		font-size: 0.875rem;
		margin-bottom: 0.25rem;
	}
	.box {
		display: flex;
		align-items: center;
		justify-content: space-between;
		min-height: 2.5rem;
		padding: 0.25rem 0.5rem;
		background: var(--bg-secondary);
		border: 1px solid transparent;
		border-radius: var(--radius);
		cursor: pointer;
	}
	.box:focus-visible {
		border-color: var(--primary);
		outline: none;
	}
	.tags {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
	}
	.tag {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		padding: 0.1rem 0.5rem;
		background: var(--primary);
		color: #fff;
		border-radius: 0.35rem;
		font-size: 0.875rem;
	}
	.tag button {
		border: none;
		background: transparent;
		color: inherit;
		cursor: pointer;
		padding: 0;
		line-height: 1;
	}
	.chevron {
		color: var(--text-muted);
	}
	.list {
		position: absolute;
		z-index: 20;
		left: 0;
		right: 0;
		margin: 0.25rem 0 0;
		padding: 0.25rem 0;
		list-style: none;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
	}
	.list button {
		width: 100%;
		text-align: left;
		padding: 0.4rem 0.75rem;
		border: none;
		background: transparent;
		cursor: pointer;
	}
	.list button:hover {
		background: var(--bg-secondary);
	}
	.empty {
		padding: 0.4rem 0.75rem;
		color: var(--text-muted);
	}
</style>
