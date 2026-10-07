'use strict';
const models = [...document.querySelectorAll('details.model')];
document.getElementById('expand-all').addEventListener('click', () => {
  models.forEach(model => { model.open = true; });
});
document.getElementById('collapse-all').addEventListener('click', () => {
  models.forEach(model => { model.open = false; });
});
// Fragment navigation opens the requested model but all six start closed without a fragment.
function openFragment() {
  const target = document.getElementById(location.hash.slice(1));
  if (target && target.matches('details.model')) target.open = true;
}
window.addEventListener('hashchange', openFragment);
openFragment();
