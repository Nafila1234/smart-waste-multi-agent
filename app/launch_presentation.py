# ==============================================================================
# ONE-CLICK PRESENTATION RUNNER (SMART WASTE MULTI-AGENT SYSTEM)
# ==============================================================================
import os
import time

print("[1/3] Installing dependencies...")
os.system("pip install -q streamlit folium groq python-dotenv joblib torch torchvision")

print("[2/3] Setting up Cloudflare tunnel binary...")
os.system("wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64")
os.system("chmod +x /usr/local/bin/cloudflared")

print("[3/3] Launching Streamlit & Opening Public URL...")
os.system("pkill -f streamlit")
os.system("pkill -f cloudflared")
os.system("streamlit run app/main.py &>/dev/null &")

time.sleep(3)
print("\n" + "="*60)
print("  🚀 APPLICATION IS READY!")
print("  Click the URL ending with '.trycloudflare.com' below:")
print("="*60 + "\n")

os.system("cloudflared tunnel --url http://localhost:8501")
