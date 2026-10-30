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
		class:open
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
			{:else}
				<span class="placeholder">Choisir des modèles…</span>
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
				<li class="empty">Tous les modèles sont sélectionnés</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.field {
		position: relative;
	}
	.label {
		display: block;
		font-size: 0.82rem;
		font-weight: 550;
		color: var(--text-muted);
		margin-bottom: 0.35rem;
	}
	.box {
		display: flex;
		align-items: center;
		justify-content: space-between;
		min-height: 2.8rem;
		padding: 0.3rem 0.7rem;
		background: var(--bg-secondary);
		border: 1px solid transparent;
		border-radius: var(--radius);
		cursor: pointer;
		transition: border-color 0.15s, box-shadow 0.15s;
	}
	.box:focus-visible,
	.box.open {
		border-color: var(--primary);
		box-shadow: 0 0 0 3px var(--primary-soft);
		outline: none;
	}
	.tags {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
	}
	.tag {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.15rem 0.3rem 0.15rem 0.65rem;
		background: var(--primary-soft);
		color: var(--primary);
		border-radius: 999px;
		font-size: 0.85rem;
		font-weight: 600;
	}
	.tag button {
		display: grid;
		place-items: center;
		width: 1.15rem;
		height: 1.15rem;
		border: none;
		border-radius: 50%;
		background: transparent;
		color: inherit;
		cursor: pointer;
		line-height: 1;
	}
	.tag button:hover {
		background: var(--primary);
		color: #fff;
	}
	.placeholder {
		color: var(--text-muted);
	}
	.chevron {
		color: var(--text-muted);
	}
	.list {
		position: absolute;
		z-index: 20;
		left: 0;
		right: 0;
		margin: 0.35rem 0 0;
		padding: 0.35rem;
		list-style: none;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		box-shadow: var(--shadow-lg);
	}
	.list button {
		width: 100%;
		text-align: left;
		padding: 0.5rem 0.7rem;
		border: none;
		border-radius: 0.5rem;
		background: transparent;
		cursor: pointer;
	}
	.list button:hover {
		background: var(--primary-soft);
		color: var(--primary);
	}
	.empty {
		padding: 0.5rem 0.7rem;
		color: var(--text-muted);
	}
</style>
