# Calibre-Web YAML 配置指南

本项目已扩展支持通过 YAML 配置文件进行配置管理，这比传统的数据库配置或命令行参数更加直观和易于维护。

## 📋 目录

- [快速开始](#快速开始)
- [配置文件结构](#配置文件结构)
- [配置优先级](#配置优先级)
- [常用配置示例](#常用配置示例)
- [环境变量](#环境变量)
- [扩展开发](#扩展开发)

## 🚀 快速开始

### 1. 使用启动脚本（推荐）

```bash
# 基本启动
./start.sh

# 使用自定义配置文件
./start.sh --config /path/to/your/config.yaml

# 开发模式
./start.sh --dev
```

### 2. 手动启动

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装 PyYAML（首次运行需要）
pip install PyYAML

# 指定配置文件并启动（可选）
export CALIBRE_CONFIG_FILE=/path/to/config.yaml
python cps.py
```

## 📁 配置文件结构

配置文件默认位于项目根目录的 `config.yaml`。完整的配置模板请查看该文件。

### 主要配置节

```yaml
server:         # 服务器相关配置
  host: 0.0.0.0
  port: 8083
  ssl: {...}    # HTTPS 配置

calibre:        # Calibre 数据库配置
  library_path: ./library

app:            # 应用设置
  title: "Calibre-Web"
  books_per_page: 60
  anonymous_browse: false

logging:        # 日志配置
  level: INFO
  access_log: {...}

mail:           # 邮件服务配置
  server: smtp.gmail.com
  port: 587

auth:           # 认证配置
  type: 0       # 0=标准, 1=LDAP

kobo:           # Kobo 设备同步
  sync: false

services:       # 外部服务
  goodreads: {...}
  google_books: {...}

converters:     # 转换工具路径
  calibre_path: /usr/bin/ebook-convert

rate_limit:     # 访问速率限制
  enabled: true

advanced:       # 高级选项
  unicode_filename: false

custom:         # 自定义配置（扩展用）
  your_feature: value
```

## 🔧 配置优先级

配置的优先级从高到低：

1. **YAML 配置文件** ← 最高优先级（覆盖其他配置）
2. 命令行参数
3. 环境变量
4. 数据库配置
5. 默认值

示例：
```bash
# YAML 中设置 port: 8083
# 命令行参数会被忽略，最终使用 8083
python cps.py  # 实际端口: 8083
```

## 💡 常用配置示例

### 1. 修改端口和监听地址

```yaml
server:
  host: 0.0.0.0    # 监听所有网卡
  port: 9090       # 自定义端口
```

### 2. 启用 HTTPS

```yaml
server:
  ssl:
    cert_file: /path/to/cert.pem
    key_file: /path/to/key.pem
```

### 3. 配置 Calibre 数据库

```yaml
calibre:
  # 相对路径（相对于项目根目录）
  library_path: ./library
  
  # 或绝对路径
  # library_path: /home/user/books/calibre
```

### 4. 启用匿名浏览和公开注册

```yaml
app:
  anonymous_browse: true
  public_registration: true
  allow_upload: true
```

### 5. 配置邮件服务（发送电子书到 Kindle）

```yaml
mail:
  server: smtp.gmail.com
  port: 587
  use_ssl: true
  login: your-email@gmail.com
  password: your-app-password
  from: "Calibre-Web <your-email@gmail.com>"
  size_limit: 26214400  # 25MB
```

### 6. 配置日志

```yaml
logging:
  level: DEBUG          # DEBUG, INFO, WARNING, ERROR
  file: /var/log/calibre-web/app.log
  access_log:
    enabled: true
    file: /var/log/calibre-web/access.log
```

### 7. LDAP 认证

```yaml
auth:
  type: 1  # 1 表示 LDAP

ldap:
  provider_url: ldap://ldap.example.com
  port: 389
  use_ssl: false
  username: "cn=admin,dc=example,dc=org"
  password: "admin_password"
  dn: "dc=example,dc=org"
  user_object: "uid=%s"
  group_name: "calibreweb"
```

### 8. 配置外部转换工具

```yaml
converters:
  calibre_path: /usr/bin/ebook-convert
  kepubify_path: /opt/kepubify/kepubify
  unrar_path: /usr/bin/unrar
```

### 9. Redis 速率限制

```yaml
rate_limit:
  enabled: true
  storage_uri: "redis://localhost:6379"
  storage_options: ""
```

## 🌍 环境变量

除了 YAML 配置，你还可以使用环境变量：

```bash
# 指定配置文件路径
export CALIBRE_CONFIG_FILE=/path/to/config.yaml

# 指定数据库路径
export CALIBRE_DBPATH=/path/to/data

# 指定端口（如果 YAML 未配置）
export CALIBRE_PORT=8080

# 启用 Flask 调试模式
export FLASK_DEBUG=1

# 自定义 Cookie 前缀
export COOKIE_PREFIX="myapp_"
```

## 🔌 扩展开发

### 添加自定义配置

1. 在 `config.yaml` 中添加自定义配置：

```yaml
custom:
  my_feature_enabled: true
  my_api_key: "secret-key"
  my_settings:
    option1: value1
    option2: value2
```

2. 在代码中读取自定义配置：

```python
from cps.config_loader import get_yaml_loader

# 获取配置加载器
loader = get_yaml_loader()

# 读取单个值
my_feature = loader.get('custom', 'my_feature_enabled', default=False)

# 读取嵌套配置
option1 = loader.get('custom', 'my_settings', 'option1')

# 获取整个 custom 节
custom_config = loader.get_custom_config()
```

### 扩展配置加载器

如果需要添加新的配置项到系统配置，可以修改 `cps/config_loader.py` 中的 `apply_to_config()` 方法：

```python
def apply_to_config(self, config):
    # ... 现有代码 ...
    
    # 添加你的自定义配置
    my_setting = self.get('my_section', 'my_setting')
    if my_setting is not None:
        config.my_custom_setting = my_setting
        log.info(f"Applied custom setting: {my_setting}")
```

## 📝 注意事项

1. **YAML 语法**：确保 YAML 文件格式正确，使用空格缩进（不要用 Tab）
2. **路径配置**：
   - 相对路径相对于项目根目录
   - 推荐使用绝对路径避免混淆
3. **密码安全**：
   - 不要将包含敏感信息的 `config.yaml` 提交到版本控制
   - 可以创建 `config.example.yaml` 作为模板
4. **配置生效**：YAML 配置会覆盖数据库配置，但 Web UI 修改后仍会保存到数据库
5. **依赖要求**：需要安装 `PyYAML`：`pip install PyYAML`

## 🛠️ 故障排除

### 配置未生效

1. 检查 YAML 文件路径是否正确
2. 查看启动日志，确认配置文件被加载
3. 确认配置项的缩进和语法正确

### YAML 解析错误

```bash
# 验证 YAML 语法
python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### PyYAML 未安装

```bash
pip install PyYAML
```

## 📚 更多资源

- [Calibre-Web 官方文档](https://github.com/janeczku/calibre-web/wiki)
- [YAML 语法指南](https://yaml.org/spec/1.2/spec.html)
- [Calibre 电子书管理](https://calibre-ebook.com/)

---

**提示**：首次运行建议使用默认配置，然后根据需要逐步修改配置文件。

