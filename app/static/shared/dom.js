export function byId(id, root = document) {
  return root.getElementById(id);
}

export function all(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

export function clearNode(node) {
  if (!node) return;
  node.replaceChildren();
}
