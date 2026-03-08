import os
import time

# 1. File persistence check
if not os.path.exists('app.py') or not os.path.exists('macroscopic_surrogate_models.pkl'):
    print("WARNING: Files missing! Colab runtime restart ho gaya tha.")
    print("Aapko Phase 1, Phase 2, aur app.py wala code cell dobara run karna padega.")
else:
    print("Files verified. Restarting servers...\n")

    # 2. Kill old hanging processes
    os.system('pkill -f streamlit')
    os.system('pkill -f cloudflared')

    # 3. Start Streamlit and log output
    os.system('nohup streamlit run app.py > streamlit_logs.txt 2>&1 &')

    print("Waiting 10 seconds for Streamlit to initialize...")
    time.sleep(10)

    # 4. Print logs to catch any Python errors
    print("\n--- Streamlit Startup Logs ---")
    os.system('head -n 10 streamlit_logs.txt')

    # 5. Start Cloudflare Tunnel
    print("\nStarting Cloudflare Tunnel...")
    os.system('nohup ./cloudflared tunnel --url http://localhost:8501 > tunnel.log 2>&1 &')
    time.sleep(8)

    print("\n=== NEW DIRECT LINK ===")
    os.system("grep -o 'https://.*\.trycloudflare\.com' tunnel.log")