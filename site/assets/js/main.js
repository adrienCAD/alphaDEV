// Renders pipeline stage cards from outputs/web/stages.json; Tech/Operator mode toggle.
const toggle = document.getElementById('mode-toggle');
const stagesEl = document.getElementById('pipeline');
let mode = 'tech';

toggle?.addEventListener('click', () => {
  mode = mode === 'tech' ? 'operator' : 'tech';
  toggle.textContent = `Mode: ${mode === 'tech' ? 'Tech' : 'Operator'}`;
  toggle.setAttribute('aria-pressed', String(mode === 'operator'));
  render();
});

let stages = [];
async function load() {
  try {
    const res = await fetch('assets/data/stages.json');
    stages = await res.json();
  } catch { stages = []; }
  render();
}

function render() {
  if (!stagesEl) return;
  stagesEl.innerHTML = stages.map(s => `
    <article class="stage">
      <h3>${s.title}</h3>
      <p>${mode === 'tech' ? s.tech : s.operator}</p>
    </article>`).join('');
}
load();
