// src/api.js

const API_BASE = "http://127.0.0.1:8000/api";

export async function searchStations(query) {
    const res = await fetch(`${API_BASE}/stations?query=${encodeURIComponent(query)}`);
    return res.json();
}

export async function fetchIntegrated(fromId, toId) {
    const res = await fetch(`${API_BASE}/integrated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            from_station_id: fromId,
            to_station_id: toId
        }),
    });

    if (!res.ok) {
        const msg = await res.text();
        throw new Error(`Integrated request failed: ${res.status} ${msg}`);
    }

    return res.json();
}