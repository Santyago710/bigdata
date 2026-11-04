"""MyTwin · servidor web (solo librería estándar + engine.py).
Uso local:  python server.py --data ./app_data --port 8000   y abrir http://localhost:8000
En Colab:   import server; server.start(DATA_DIR, port=8000)
"""
import json, argparse, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
from engine import Engine

HERE = Path(__file__).resolve().parent


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x); return None if (x != x or x in (float('inf'), float('-inf'))) else x
    return x


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, body, code=200, ctype='application/json; charset=utf-8'):
            if not isinstance(body, bytes): body = body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(clean(obj), ensure_ascii=False), code)

        def do_GET(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ('/', '/index.html'):
                    return self._send((HERE / 'static' / 'index.html').read_bytes(), ctype='text/html; charset=utf-8')
                if u.path == '/api/info': return self._json(engine.info())
                if u.path == '/api/users':
                    return self._json(engine.sample_users(int(q.get('n', 10)), int(q['seed']) if 'seed' in q else None, q.get('q')))
                if u.path == '/api/search': return self._json(engine.search(q.get('q', '')))
                if u.path == '/api/popular': return self._json(engine.popular(q.get('platform'), int(q.get('n', 12))))
                self._json({'error': 'no encontrado'}, 404)
            except Exception as e:
                traceback.print_exc(); self._json({'error': str(e)}, 500)

        def do_POST(self):
            try:
                n = int(self.headers.get('Content-Length', 0)); req = json.loads(self.rfile.read(n) or b'{}')
                if urlparse(self.path).path == '/api/recommend': return self._json(engine.recommend(req))
                self._json({'error': 'no encontrado'}, 404)
            except Exception as e:
                traceback.print_exc(); self._json({'error': str(e)}, 500)
    return H


def start(data_dir, port=8000, host='127.0.0.1', background=True):
    engine = Engine(data_dir)
    srv = ThreadingHTTPServer((host, port), make_handler(engine))
    print(f'MyTwin listo: {engine.n_users:,} usuarios · {engine.n_items:,} productos -> http://{host}:{port}')
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    else:
        srv.serve_forever()
    return srv


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    a = ap.parse_args(); start(a.data, a.port, a.host, background=False)
