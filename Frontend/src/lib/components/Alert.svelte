<script lang="ts">
	import type { Snippet } from 'svelte';
	import Icon from './Icon.svelte';

	let { kind, children }: { kind: 'success' | 'error' | 'warning' | 'info'; children: Snippet } = $props();

	const icon = { success: 'check', error: 'alert', warning: 'alert', info: 'info' } as const;
</script>

<div class="alert {kind}" role="alert">
	<Icon name={icon[kind]} size={18} />
	<div>{@render children()}</div>
</div>

<style>
	.alert {
		display: flex;
		align-items: flex-start;
		gap: 0.65rem;
		padding: 0.8rem 1rem;
		border-radius: var(--radius);
		border: 1px solid color-mix(in srgb, var(--c) 30%, transparent);
		background: var(--bg-c);
		color: var(--c);
		font-weight: 500;
	}
	.alert :global(svg) {
		margin-top: 0.2rem;
	}
	.success {
		--c: var(--success);
		--bg-c: var(--success-bg);
	}
	.error {
		--c: var(--error);
		--bg-c: var(--error-bg);
	}
	.warning {
		--c: var(--warning);
		--bg-c: var(--warning-bg);
	}
	.info {
		--c: var(--info);
		--bg-c: var(--info-bg);
	}
</style>
