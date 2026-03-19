const net = require('net');

function forward(fromPort, toPort, label) {
  net.createServer(s => {
    const t = net.connect(toPort, '127.0.0.1', () => { s.pipe(t); t.pipe(s); });
    t.on('error', () => s.destroy());
    s.on('error', () => t.destroy());
  }).listen(fromPort, '0.0.0.0', () => console.log(`[relay] ${label}: 0.0.0.0:${fromPort} -> 127.0.0.1:${toPort}`));
}

forward(18799, 18789, 'gateway');
forward(18801, 18790, 'portal');
