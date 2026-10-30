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
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.9rem;
	}
	th {
		text-align: left;
		font-size: 0.74rem;
		font-weight: 650;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--text-muted);
		background: var(--bg-secondary);
		white-space: nowrap;
	}
	th,
	td {
		padding: 0.65rem 0.9rem;
		border-bottom: 1px solid var(--border);
	}
	tbody tr:last-child td {
		border-bottom: none;
	}
	tbody tr:hover {
		background: var(--primary-soft);
	}
	td:first-child {
		font-weight: 600;
	}
</style>
