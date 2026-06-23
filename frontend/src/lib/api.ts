const API_URL = "/api/v1";

export async function fetchChartData(endpoint: string) {
  const res = await fetch(`${API_URL}${endpoint}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch chart data: ${res.statusText}`);
  }
  const data = await res.json();
  return data;
}

export async function fetchJsonData(endpoint: string) {
  const res = await fetch(`${API_URL}${endpoint}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch JSON data: ${res.statusText}`);
  }
  return res.json();
}
