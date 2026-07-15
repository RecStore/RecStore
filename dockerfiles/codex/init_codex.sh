#!/bin/bash
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
cd "$SCRIPT_DIR"
set -x
set -e
# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
bash install.sh
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

nvm install --lts
npm install -g @openai/codex

curl https://cursor.com/install -fsS | bash
curl -fsSL https://claude.ai/install.sh | bash
curl -fsSL https://github.com/SaladDay/cc-switch-cli/releases/latest/download/install.sh | bash

cp -r .cc-switch ~/

echo 'root:1234' | chpasswd
sed -Ei 's/#PermitRootLogin.*/PermitRootLogin yes/g' /etc/ssh/sshd_config

service ssh restart 

git config --global user.name "Minhui Xie"
git config --global user.email "645214784@qq.com"

npm i -g happy
npm install -g @getpaseo/cli
