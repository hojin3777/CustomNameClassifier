const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const { create } = require('domain');

let mainWindow;
let pythonProcess;

// ****** Python 백엔드 관리 ******
// startPythonBackend: Python 백엔드 프로세스 시작
function startPythonBackend() {
    const isDev = !app.isPackaged;

    let pythonPath;
    let scriptPath;

    if (isDev) {
        // ✨ 개발 모드
        pythonPath = 'C:\\code\\.venv_backend\\Scripts\\python.exe';
        scriptPath = path.join(__dirname, '..', '..', '..', 'customMydataService', 'backend', 'app.py');
    } else {
        // ✨ 프로덕션 모드: 패키징된 Python 사용
        pythonPath = path.join(process.resourcesPath, 'python', 'Scripts', 'python.exe');
        scriptPath = path.join(process.resourcesPath, 'backend', 'app.py');
    }

    console.log('Starting Python backend...');
    console.log('Python path:', pythonPath);
    console.log('Script path:', scriptPath);

    // ✨ 경로 검증
    if (!fs.existsSync(pythonPath)) {
        console.error('❌ Python executable not found at:', pythonPath);
        dialog.showErrorBox('Python 오류', `Python 실행 파일을 찾을 수 없습니다:\n${pythonPath}`);
        return;
    }

    if (!fs.existsSync(scriptPath)) {
        console.error('❌ app.py not found at:', scriptPath);
        dialog.showErrorBox('백엔드 오류', `백엔드 스크립트를 찾을 수 없습니다:\n${scriptPath}`);
        return;
    }

    const spawnOptions = {
        cwd: path.dirname(scriptPath),
        env: {
            ...process.env,
            // ✨ Python 경로 설정
            PYTHONPATH: isDev
                ? 'C:\\code\\pororo_easyocr_main'
                : path.join(process.resourcesPath, 'pororo_easyocr_main'),

            // ✨ 리소스 경로 환경 변수로 전달
            RESOURCE_PATH: isDev
                ? 'C:\\code'
                : process.resourcesPath,

            // ✨ 개발/프로덕션 모드 전달
            IS_PACKAGED: isDev ? 'false' : 'true'
        }
    };

    pythonProcess = spawn(pythonPath, [scriptPath], spawnOptions);

    pythonProcess.stdout.on('data', (data) => {
        console.log(`🐍 Python: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`🐍 Python Error: ${data}`);
    });

    pythonProcess.on('error', (err) => {
        console.error('❌ Failed to start Python process:', err);
        dialog.showErrorBox('백엔드 시작 실패', `Python 프로세스를 시작할 수 없습니다:\n${err.message}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`🐍 Python process exited with code ${code}`);
    });
}


async function waitForBackend(maxRetries = 180, interval = 1000) {
    console.log('⏳ Waiting for backend to be ready...');

    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await fetch('http://localhost:5000/api/health');
            if (response.ok) {
                console.log('Backend is ready!');
                return true;
            }
        } catch (error) {
            if (i % 10 === 0) {
                console.log(`Backend not ready yet (${i + 1}/${maxRetries})...`);
            }
        }

        // interval만큼 대기
        await new Promise(resolve => setTimeout(resolve, interval));
    }

    console.error('Backend failed to start within timeout');
    return false;
}

// createWindow: 메인 윈도우 생성
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1600,
        height: 900,
        minWidth: 1250,
        minHeight: 800,
        frame: false,
        backgroundColor: '#1a1a1a',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            enableRemoteModule: false,
            preload: path.join(__dirname, 'preload.js')
        },
        icon: path.join(__dirname, '..', 'public', 'icon.ico'),
        show: false,
        titleBarStyle: 'hidden',
        trafficLightPosition: { x: 10, y: 10 }
    });

    const isDev = !app.isPackaged;
    const startUrl = isDev
        ? 'http://localhost:5173'
        : `file://${path.join(__dirname, '..', 'dist', 'index.html')}`;

    mainWindow.loadURL(startUrl);

    mainWindow.webContents.on('did-finish-load', async () => {
        console.log('✅ Frontend loaded');
        mainWindow.show();
        if (isDev) {
            mainWindow.webContents.openDevTools();
        }
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// ****** 앱 생명주기 관리 ******
app.whenReady().then(() => {
    startPythonBackend();
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    dialog.showErrorBox('오류 발생', error.message);
});

// ****** 윈도우 제어 IPC 핸들러 ******
ipcMain.on('window-minimize', () => {
    if (mainWindow) mainWindow.minimize();
});

ipcMain.on('window-maximize', () => {
    if (mainWindow) {
        if (mainWindow.isMaximized()) {
            mainWindow.unmaximize();
        } else {
            mainWindow.maximize();
        }
    }
});

ipcMain.on('window-close', () => {
    if (mainWindow) mainWindow.close();
});