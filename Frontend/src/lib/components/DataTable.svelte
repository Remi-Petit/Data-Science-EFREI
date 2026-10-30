<script lang="ts">
	let { columns, rows }: { columns: string[]; rows: (string | number | null)[][] } = $props();
</script>

<div class="wrap">
	<table>
		<thead>
			<tr>{#each columns as c (c)}<th>{c}</th>{/each}</tr>
		</thead>
		<tbody>
			{#each rows as row, i (i)}
				<tr>{#each row as cell, j (j)}<td>{cell ?? '–'}</td>{/each}</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.wrap {
		overflow-x: auto;
		border: 1px solid var(--border);
		border-radius: var(--radius);
		margin-bottom: 1rem;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.9rem;
	}
	th {
		text-align: left;
		font-weight: 600;
		background: var(--bg-secondary);
		white-space: nowrap;
	}
	th,
	td {
		padding: 0.5rem 0.75rem;
		border-bottom: 1px solid rgba(49, 51, 63, 0.1);
	}
	tbody tr:last-child td {
		border-bottom: none;
	}
</style>
