// loopmm desktop app: serves the chat page and the bundled model over app:// (with cross-origin isolation, so
// onnxruntime-web can use threads) and opens one window. Nothing goes over the network.
const { app, BrowserWindow, protocol, shell } = require('electron');
const path = require('path');
const fs = require('fs');

protocol.registerSchemesAsPrivileged([{ scheme: 'app',
  privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } }]);

const WEB = path.join(__dirname, 'web');
const MODELS = app.isPackaged ? path.join(process.resourcesPath, 'models') : path.join(__dirname, 'models');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.png': 'image/png' };

function filePath(pathname) {
  const p = decodeURIComponent(pathname).replace(/^\/+/, '') || 'index.html';
  const [root, rel] = p.startsWith('models/') ? [MODELS, p.slice(7)] : [WEB, p];
  const full = path.normalize(path.join(root, rel));
  if (!full.startsWith(root)) throw new Error('bad path');
  return full;
}

function createWindow() {
  const smoke = process.env.LOOPMM_SMOKE;
  const win = new BrowserWindow({ width: 1100, height: 780, minWidth: 380, minHeight: 400, title: 'loopmm',
    backgroundColor: '#161615', show: !smoke, webPreferences: { contextIsolation: true, sandbox: true } });
  win.webContents.setWindowOpenHandler(({ url }) => { if (/^https?:/.test(url)) shell.openExternal(url); return { action: 'deny' }; });
  win.webContents.on('will-navigate', (e, url) => { if (!url.startsWith('app://')) { e.preventDefault(); shell.openExternal(url); } });
  if (smoke) {
    win.webContents.on('console-message', (_e, _l, msg) => console.log('[page]', msg));
    win.on('page-title-updated', (_e, title) => {
      if (title.startsWith('SMOKE:')) { fs.writeFileSync(smoke, title.slice(6)); console.log(title); app.exit(0); }
    });
    setTimeout(() => { console.log('smoke timeout'); app.exit(2); }, 15 * 60 * 1000);
  }
  win.loadURL('app://loopmm/index.html' + (smoke ? '?smoke=1' : ''));
}

app.whenReady().then(() => {
  protocol.handle('app', async (req) => {
    try {
      const f = filePath(new URL(req.url).pathname);
      const data = await fs.promises.readFile(f);
      return new Response(data, { headers: { 'content-type': TYPES[path.extname(f)] || 'application/octet-stream',
        'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp',
        'cross-origin-resource-policy': 'same-origin' } });
    } catch (e) {
      return new Response('not found', { status: 404 });
    }
  });
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});
app.on('window-all-closed', () => app.quit());
