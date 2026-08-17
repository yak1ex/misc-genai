param(
    [switch]$ShowOllama,
    [switch]$ShowWebUI,
    [switch]$UpdateOllama,
    [switch]$UpdateWebUI
)

if (-not $ShowOllama -and -not $ShowWebUI -and -not $UpdateOllama -and -not $UpdateWebUI) {
    # FIXME: Now, there are many containers, so this is not a good idea to show all of them. We should only show the containers that we care about.
    wsl -e docker container list -a
    Exit
}

if ($ShowOllama) {
    wsl -e docker start ollama
    wsl -e docker exec -it ollama ollama -v
}

if ($ShowWebUI) {
    wsl -e docker start open-webui
    wsl -e docker exec -it open-webui cat /app/package.json | jq .version
}

if ($UpdateOllama) {
    wsl -e docker stop ollama
    wsl -e docker rm ollama
    wsl -e docker run -d --gpus=all -v ollama:/root/.ollama --pull always --restart unless-stopped -p 11434:11434 --name ollama ollama/ollama
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
        '--restart', 'unless-stopped',
        'ghcr.io/open-webui/open-webui:cuda')
}
