#!/bin/bash

# NFL Player Trajectory Prediction - GitHub 上传脚本
# 使用方法：在终端执行 bash upload.sh

cd /Users/yuanzhiyi/Desktop/nfl-player-trajectory-prediction

echo "正在配置 Git..."
git config user.name "YZY0108"
git config user.email "your.email@example.com"  # 请替换为你的邮箱

echo "正在推送到 GitHub..."
git remote set-url origin https://github.com/YZY0108/nfl-player-trajectory-prediction.git

echo "请在浏览器中完成认证..."
git push -u origin main

echo "完成！"
echo "访问: https://github.com/YZY0108/nfl-player-trajectory-prediction"

