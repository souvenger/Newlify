mkdir -p ~/.streamlit/
echo "[theme]
primaryColor='#e41212'
backgroundColor='#f1b9b1'
secondaryBackgroundColor='#b7130b'
font = 'monospace'
[server]
headless = true
port = $PORT
enableCORS = true
" > ~/.streamlit/config.toml