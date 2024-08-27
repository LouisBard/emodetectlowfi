# Emodetectlowfi

Pour le raspi il faut créer un environnement virtuel:

python3 -m venv emotion_env
source emotion_env/bin/activate

## Deployment

On Raspberry Pi 5 to enable the systemd service:
- git pull origin main
- sudo cp artboot.service /etc/systemd/system/artboot.service
- sudo systemctl daemon-reload
- sudo systemctl enable artboot.service
- sudo reboot

In case of errors inspect:
- sudo systemctl status artboot.service
- sudo journalctl -u artboot.service -f
