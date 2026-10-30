<script lang="ts">
	import type { Snippet } from 'svelte';

	let { kind, children }: { kind: 'success' | 'error' | 'warning' | 'info'; children: Snippet } = $props();
</script>

<div class="alert {kind}" role="alert">{@render children()}</div>

<style>
	.alert {
		padding: 1rem;
		border-radius: var(--radius);
		margin-bottom: 1rem;
	}
	.success {
		background: var(--success-bg);
		color: var(--success-fg);
	}
	.error {
		background: var(--error-bg);
		color: var(--error-fg);
	}
	.warning {
		background: var(--warning-bg);
		color: var(--warning-fg);
	}
	.info {
		background: var(--info-bg);
		color: var(--info-fg);
	}
</style>
