import { spawn } from 'child_process';

const file = process.argv[2] || '';
const openPath = file ? `/?file=${file}` : '/';

spawn('vite', ['--open', openPath], { stdio: 'inherit', shell: true });
