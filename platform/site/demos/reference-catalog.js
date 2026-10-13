const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
};

function sourceLinks(sources) {
  const links = el('div', null, 'reference-sources');
  for (const source of sources) {
    const link = el('a', source.label);
    link.href = source.url;
    link.target = '_blank';
    link.rel = 'noreferrer';
    links.append(link);
  }
  return links;
}

function renderRows(target, rows, fields) {
  document.querySelector(target).replaceChildren(...rows.map((row) => {
    const tr = el('tr');
    tr.dataset.referenceId = row.id;
    const identity = el('td');
    identity.append(el('strong', row.name), el('small', row.architecture || row.role || row.kind, 'reference-category'), sourceLinks(row.sources));
    tr.append(identity);
    for (const field of fields) {
      const td = el('td', row[field]);
      if (field === 'release') td.append(el('small', row.license, 'reference-category'));
      tr.append(td);
    }
    return tr;
  }));
}

try {
  const response = await fetch('./data/lyrics-reference.json?v=20261013');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  for (const date of document.querySelectorAll('[data-lyrics-reference-date]')) date.textContent = `官方来源核验 ${data.checked_at}`;
  const renderModels = (family) => {
    const rows = family === 'all' ? data.models : data.models.filter((row) => row.family === family);
    renderRows('#lyrics-model-reference-body', rows, ['input', 'release', 'posttrain']);
    document.querySelector('#lyrics-model-reference-count').textContent = `${rows.length} / ${data.models.length} models`;
    for (const button of document.querySelectorAll('[data-model-family]')) button.setAttribute('aria-pressed', String(button.dataset.modelFamily === family));
  };
  for (const button of document.querySelectorAll('[data-model-family]')) button.addEventListener('click', () => renderModels(button.dataset.modelFamily));
  renderModels('all');
  renderRows('#lyrics-reward-reference-body', data.rewards, ['definition', 'use', 'risk']);
  renderRows('#lyrics-dataset-reference-body', data.datasets, ['content', 'protocol', 'boundary']);
} catch (error) {
  for (const target of ['#lyrics-model-reference-body', '#lyrics-reward-reference-body', '#lyrics-dataset-reference-body']) {
    const td = el('td', `调研资料加载失败 (${error.message})。下方 GitHub 源清单仍可查阅。`);
    td.colSpan = 4;
    const tr = el('tr');
    tr.append(td);
    document.querySelector(target).replaceChildren(tr);
  }
}
