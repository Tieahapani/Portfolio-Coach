// React stats strip for the Progress page.
// Loaded via CDN React + Babel standalone — no build step.

function StatCard({ label, value, tone }) {
  return (
    <div className={'stat-card' + (tone ? ' ' + tone : '')}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function ProgressStats({ projects }) {
  const count = (status) => projects.filter((p) => p.status === status).length;
  const stats = {
    total: projects.length,
    active: count('active'),
    stalled: count('stalled'),
    completed: count('completed'),
    commits: projects.reduce((n, p) => n + (p.commit_count || 0), 0),
  };

  if (!stats.total) return null;

  return (
    <div className="stats-strip">
      <StatCard label="Tracked" value={stats.total} />
      <StatCard label="Active" value={stats.active} tone="good" />
      <StatCard label="Stalled" value={stats.stalled} tone={stats.stalled ? 'warn' : ''} />
      <StatCard label="Completed" value={stats.completed} tone="accent" />
      <StatCard label="Total commits" value={stats.commits} />
    </div>
  );
}

const statsRoot = ReactDOM.createRoot(document.getElementById('statsRoot'));

window.renderProgressStats = (projects) => {
  statsRoot.render(<ProgressStats projects={projects || []} />);
};

// If the page loaded projects before Babel finished transpiling this file,
// render the stashed data now.
if (window._statsData) {
  window.renderProgressStats(window._statsData);
}
