# Matplotlib 字体警告说明

## 警告信息含义

您看到的警告信息：

```
UserWarning: Glyph 23458 (\N{CJK UNIFIED IDEOGRAPH-5BA2}) missing from current font.
```

**含义**：
- matplotlib 在生成图表时使用了中文字符（如"客户端资源使用时间线"、"服务端资源监控"等）
- 但是当前 matplotlib 使用的字体不支持这些中文字符（CJK 统一表意文字）
- matplotlib 会显示警告，但图表仍然会生成（中文可能显示为方块或缺失）

## 警告中的中文字符

警告中提到的 Unicode 码点对应的中文字符：
- `23458` = 客
- `25143` = 户
- `31471` = 端
- `36164` = 资
- `28304` = 源
- `20351` = 使
- `29992` = 用
- `26102` = 时
- `38388` = 间
- `32447` = 线
- `26381` = 服
- `21153` = 务

这些字符出现在图表标题中，如：
- "客户端资源使用时间线"
- "服务端资源使用时间线"

## 已实施的修复

### 1. 改进字体配置（已修复）

已修改 `SplitLearnMonitor/src/splitlearn_monitor/visualizers/time_series_viz.py`：

- ✅ **自动检测系统中可用的中文字体**
- ✅ **优先使用系统默认中文字体**（macOS 使用 PingFang SC，Windows 使用 Microsoft YaHei）
- ✅ **在每次绘图前确保字体配置生效**

### 2. 字体配置策略

代码会根据操作系统自动选择合适的中文字体：

**macOS：**
- PingFang SC（首选）
- PingFang TC
- STHeiti
- Heiti SC/TC

**Windows：**
- Microsoft YaHei（首选）
- SimHei
- SimSun

**Linux：**
- WenQuanYi Micro Hei
- Noto Sans CJK SC

## 如果警告仍然出现

### 方法 1：抑制警告（不影响功能）

如果警告不影响使用，可以抑制警告：

```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
```

### 方法 2：检查字体安装

在 macOS 上，确保中文字体已安装：

```bash
# 检查字体
fc-list :lang=zh

# 或使用 Python
python3 -c "import matplotlib.font_manager as fm; fonts = [f.name for f in fm.fontManager.ttflist if 'PingFang' in f.name or 'Heiti' in f.name]; print(fonts)"
```

### 方法 3：手动指定字体

如果自动检测失败，可以手动指定：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'sans-serif']
```

## 影响

- **功能影响**：无。图表仍然可以正常生成和使用
- **显示影响**：如果字体配置失败，中文可能显示为方块，但不影响数据查看
- **性能影响**：无

## 总结

这些警告是**信息性警告**，不影响功能。代码已经实现了：
1. ✅ 自动检测系统中文字体
2. ✅ 在绘图前确保字体配置生效
3. ✅ 跨平台字体支持（macOS/Windows/Linux）

如果仍有警告，可以安全忽略，或使用警告过滤器抑制。
