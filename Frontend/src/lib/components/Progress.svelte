<script lang="ts">
	let { value, text }: { value: number; text: string } = $props();
</script>

<div class="progress">
	<div class="track"><div class="fill" style="width: {value * 100}%"></div></div>
	<div class="caption">{text}</div>
</div>

<style>
	.track {
		height: 0.5rem;
		margin-bottom: 0.4rem;
		background: var(--bg-secondary);
		border-radius: 999px;
		overflow: hidden;
	}
	.fill {
		height: 100%;
		background: var(--gradient);
		border-radius: 999px;
		transition: width 0.25s;
	}
</style>
