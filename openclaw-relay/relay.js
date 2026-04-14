const net  = require('net');
const http = require('http');

// ── Connection counters ───────────────────────────────────────────────────────

const counters = {
  gateway: { active: 0, total: 0 },
  portal:  { active: 0, total: 0 },
};

let healthStatus = 0;   // 0 = unknown, 1 = up
let healthChecks = 0;
let healthFailures = 0;

// ── TCP forwarder ─────────────────────────────────────────────────────────────

function forward(fromPort, toPort, label) {
  const key = label;
  net.createServer(s => {
    counters[key].active++;
    counters[key].total++;
    const t = net.connect(toPort, '127.0.0.1', () => { s.pipe(t); t.pipe(s); });
    const cleanup = () => { counters[key].active = Math.max(0, counters[key].active - 1); };
    t.on('error', () => { s.destroy(); });
    s.on('error', () => { t.destroy(); });
    s.on('close', cleanup);
    t.on('close', cleanup);
  }).listen(fromPort, '::', () =>
    console.log(`[relay] ${label}: [::]:${fromPort} -> [::1]:${toPort}`)
  );
}

forward(18799, 18789, 'gateway');
forward(18801, 18790, 'portal');

// ── Health poller ─────────────────────────────────────────────────────────────

function pollHealth() {
  const req = http.request(
    { host: '::1', port: 18789, path: '/health', method: 'GET', timeout: 3000 },
    res => {
      healthChecks++;
      healthStatus = res.statusCode === 200 ? 1 : 0;
      if (healthStatus === 0) healthFailures++;
    }
  );
  req.on('error', () => { healthChecks++; healthFailures++; healthStatus = 0; });
  req.end();
}
pollHealth();
setInterval(pollHealth, 15000);

// ── Prometheus metrics endpoint on port 9091 ──────────────────────────────────

http.createServer((req, res) => {
  if (req.url !== '/metrics') {
    res.writeHead(404); res.end(); return;
  }
  const lines = [
    '# HELP openclaw_up OpenClaw gateway health (1=up)',
    '# TYPE openclaw_up gauge',
    `openclaw_up ${healthStatus}`,

    '# HELP openclaw_health_checks_total Total health checks performed',
    '# TYPE openclaw_health_checks_total counter',
    `openclaw_health_checks_total ${healthChecks}`,

    '# HELP openclaw_health_failures_total Total failed health checks',
    '# TYPE openclaw_health_failures_total counter',
    `openclaw_health_failures_total ${healthFailures}`,

    '# HELP openclaw_gateway_connections_active Active connections on gateway port',
    '# TYPE openclaw_gateway_connections_active gauge',
    `openclaw_gateway_connections_active ${counters.gateway.active}`,

    '# HELP openclaw_gateway_connections_total Total connections accepted on gateway port',
    '# TYPE openclaw_gateway_connections_total counter',
    `openclaw_gateway_connections_total ${counters.gateway.total}`,

    '# HELP openclaw_portal_connections_active Active connections on portal port',
    '# TYPE openclaw_portal_connections_active gauge',
    `openclaw_portal_connections_active ${counters.portal.active}`,

    '# HELP openclaw_portal_connections_total Total connections accepted on portal port',
    '# TYPE openclaw_portal_connections_total counter',
    `openclaw_portal_connections_total ${counters.portal.total}`,
  ];
  res.writeHead(200, { 'Content-Type': 'text/plain; version=0.0.4' });
  res.end(lines.join('\n') + '\n');
}).listen(9091, '::', () => console.log('[relay] metrics: http://[::]:9091/metrics'));
