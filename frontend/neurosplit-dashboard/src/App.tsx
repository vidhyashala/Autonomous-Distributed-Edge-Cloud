import React from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const signals = [
  { label: 'Dynamic split layer', value: 'L7 → hybrid edge/cloud' },
  { label: 'Latency forecast', value: '24 ms p50 / 41 ms p95' },
  { label: 'Teleportation', value: '3 layers prewarmed' },
  { label: 'Swarm cluster', value: '5 trusted peers' },
  { label: 'Energy objective', value: '32% reduction target' },
];

function App() {
  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">NeuroSplit-X Control Observatory</p>
        <h1>Living distributed neural organism for edge-cloud intelligence</h1>
        <p>
          Monitor adaptive split routing, neural teleportation, digital-twin forecasts,
          swarm cognition, and energy-aware orchestration in real time.
        </p>
      </section>
      <section className="grid">
        {signals.map((signal) => (
          <article className="card" key={signal.label}>
            <span>{signal.label}</span>
            <strong>{signal.value}</strong>
          </article>
        ))}
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
