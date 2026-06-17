export async function responseText(response) {
  try {
    return await response.text();
  } catch (_error) {
    return '';
  }
}

export async function responseJson(response) {
  try {
    return await response.json();
  } catch (_error) {
    return null;
  }
}

export async function readResponseError(response, label = 'request') {
  const status = response?.status ? `HTTP ${response.status}` : 'request failed';
  const text = await responseText(response);
  return `${label} failed: ${status}${text ? `: ${text}` : ''}`;
}
