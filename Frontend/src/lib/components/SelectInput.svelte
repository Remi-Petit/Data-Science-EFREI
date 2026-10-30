<script lang="ts" generics="T extends string | number">
	type Option = { value: T; label: string };

	let { label, value = $bindable(), options }: { label: string; value: T; options: Option[] } = $props();
</script>

<label class="field">
	<span class="label">{label}</span>
	<select bind:value>
		{#each options as o (o.value)}
			<option value={o.value}>{o.label}</option>
		{/each}
	</select>
</label>

<style>
	.field {
		display: block;
		min-width: 0;
	}
	.label {
		display: block;
		font-size: 0.82rem;
		font-weight: 550;
		color: var(--text-muted);
		margin-bottom: 0.35rem;
	}
	select {
		width: 100%;
		padding: 0.55rem 2.2rem 0.55rem 0.8rem;
		appearance: none;
		background-color: var(--bg-secondary);
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");
		background-repeat: no-repeat;
		background-position: right 0.7rem center;
		border: 1px solid transparent;
		border-radius: var(--radius);
		cursor: pointer;
		transition: border-color 0.15s, box-shadow 0.15s;
	}
	select:focus {
		outline: none;
		border-color: var(--primary);
		box-shadow: 0 0 0 3px var(--primary-soft);
	}
</style>
