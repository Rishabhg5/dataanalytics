const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let apiProcess;

function createWindow() {
  // 1. Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // 2. Load the React App
  // In development, load localhost. In production, load the built index.html
  const startUrl = process.env.ELECTRON_START_URL || `file://${path.join(__dirname, 'index.html')}`;
  mainWindow.loadURL(startUrl);

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

// 3. Launch the Python Backend
function startApiServer() {
  // Adjust this path to where you put the .exe
  const apiPath = path.join(process.resourcesPath, 'api_server.exe');

  // Spawn the process
  apiProcess = spawn(apiPath);

  apiProcess.stdout.on('data', (data) => {
    console.log(`API: ${data}`);
  });
}

app.on('ready', () => {
  startApiServer();
  createWindow();
});

// 4. Kill the Python Backend when App Closes
app.on('will-quit', () => {
  if (apiProcess) {
    apiProcess.kill();
  }
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});