param(
    [switch]$ShowOllama,
    [switch]$ShowWebUI,
    [switch]$UpdateOllama,
    [switch]$UpdateWebUI
)

if (-not $ShowOllama -and -not $ShowWebUI -and -not $UpdateOllama -and -not $UpdateWebUI) {
    wsl -e docker container list -a
    Exit
}

if ($ShowOllama) {
    wsl -e docker exec -it ollama ollama -v
}

if ($ShowWebUI) {
    wsl -e docker exec -it open-webui cat /app/package.json | jq .version
}

if ($UpdateOllama) {
    wsl -e docker pull ollama/ollama
    wsl -e docker stop ollama
    wsl -e docker rm ollama
    wsl -e docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
}

if ($UpdateWebUI) {
    wsl -e docker pull ghcr.io/open-webui/open-webui:cuda
    wsl -e docker stop open-webui
    wsl -e docker rm open-webui
    wsl -e @('docker', 'run', '-d', '--gpus=all',
        '-p', '3000:8080',
        '--add-host=host.docker.internal:host-gateway',
        '-v', 'open-webui:/app/backend/data',
        '--name', 'open-webui',
        '--restart', 'always',
        'ghcr.io/open-webui/open-webui:cuda')
}
