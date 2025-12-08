#!/bin/bash
# 更新其他 Python 环境的脚本

echo "=========================================="
echo "Python 环境更新工具"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 更新 Homebrew Python
echo -e "${YELLOW}1. 检查 Homebrew Python...${NC}"
if [ -f "/opt/homebrew/bin/python3" ]; then
    echo "   Homebrew Python 版本:"
    /opt/homebrew/bin/python3 --version
    echo "   更新 Homebrew Python (需要管理员权限)..."
    read -p "   是否更新 Homebrew Python? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        brew upgrade python@3.13 python@3.12 python@3.11 2>/dev/null || echo "   跳过（可能已是最新版本）"
    fi
else
    echo "   Homebrew Python 未安装"
fi
echo ""

# 2. 更新 Python.org Framework
echo -e "${YELLOW}2. 检查 Python.org Framework...${NC}"
if [ -f "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3" ]; then
    echo "   Python.framework 3.11 版本:"
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 --version
    echo "   更新 pip..."
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pip install --upgrade pip --user 2>/dev/null
    echo "   ✓ pip 已更新"
else
    echo "   Python.framework 3.11 未找到"
fi
echo ""

# 3. 更新 Anaconda Python
echo -e "${YELLOW}3. 检查 Anaconda Python...${NC}"
if [ -f "/Users/lhy/anaconda3/bin/python3" ]; then
    echo "   Anaconda Python 版本:"
    /Users/lhy/anaconda3/bin/python3 --version
    echo "   更新 Anaconda (需要一些时间)..."
    read -p "   是否更新 Anaconda Python? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        /Users/lhy/anaconda3/bin/conda update python -y 2>/dev/null || echo "   跳过（可能需要手动运行: conda update python）"
    fi
else
    echo "   Anaconda Python 未找到"
fi
echo ""

# 4. 更新系统 Python (不推荐，但可以更新 pip)
echo -e "${YELLOW}4. 检查系统 Python...${NC}"
if [ -f "/usr/bin/python3" ]; then
    echo "   系统 Python 版本:"
    /usr/bin/python3 --version
    echo "   ⚠️  不建议更新系统 Python，但可以更新 pip..."
    read -p "   是否更新系统 Python 的 pip? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        /usr/bin/python3 -m pip install --upgrade pip --user 2>/dev/null || echo "   跳过（可能需要 sudo）"
    fi
else
    echo "   系统 Python 未找到"
fi
echo ""

# 5. 显示所有 Python 环境
echo -e "${GREEN}5. 当前所有 Python 环境:${NC}"
echo ""
which -a python3 | while read python_path; do
    if [ -f "$python_path" ]; then
        version=$($python_path --version 2>&1)
        echo "   $python_path"
        echo "     版本: $version"
        echo ""
    fi
done

echo -e "${GREEN}=========================================="
echo "更新完成！"
echo "==========================================${NC}"

