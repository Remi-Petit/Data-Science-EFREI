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
		margin-bottom: 1rem;
	}
	.label {
		display: block;
		font-size: 0.875rem;
		margin-bottom: 0.25rem;
	}
	select {
		width: 100%;
		padding: 0.5rem 0.75rem;
		background: var(--bg-secondary);
		border: 1px solid transparent;
		border-radius: var(--radius);
		outline: none;
		cursor: pointer;
	}
	select:focus {
		border-color: var(--primary);
	}
</style>
