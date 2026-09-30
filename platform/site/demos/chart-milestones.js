export function createMilestones(canvas, maxStep) {
  const tooltip = document.createElement('div');
  tooltip.className = 'training-chart-tooltip';
  tooltip.hidden = true;
  canvas.parentElement.append(tooltip);
  let milestones = [];
  const x = (step, width) => 39 + (width - 51) * step / maxStep;

  canvas.addEventListener('pointermove', (event) => {
    const bounds = canvas.getBoundingClientRect();
    const localX = event.clientX - bounds.left;
    const nearest = milestones.reduce((best, item) =>
      !best || Math.abs(x(item.step, bounds.width) - localX) < Math.abs(x(best.step, bounds.width) - localX) ? item : best, null);
    if (!nearest || Math.abs(x(nearest.step, bounds.width) - localX) > 14) {
      tooltip.hidden = true;
      return;
    }
    tooltip.textContent = nearest.label;
    tooltip.style.left = `${Math.max(72, Math.min(bounds.width - 72, x(nearest.step, bounds.width)))}px`;
    tooltip.style.top = `${canvas.offsetTop + 8}px`;
    tooltip.hidden = false;
  });
  canvas.addEventListener('pointerleave', () => { tooltip.hidden = true; });

  return {
    set(items) {
      milestones = items;
      tooltip.hidden = true;
      canvas.setAttribute('aria-description', items.map((item) => item.label).join('；'));
    },
    draw(context, width, axisY, color) {
      for (const item of milestones) {
        context.beginPath();
        context.arc(x(item.step, width), axisY, 4, 0, Math.PI * 2);
        context.fillStyle = '#fff';
        context.fill();
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.stroke();
      }
    },
  };
}
