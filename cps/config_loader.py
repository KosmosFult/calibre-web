# -*- coding: utf-8 -*-

#  This file is part of the Calibre-Web (https://github.com/janeczku/calibre-web)
#    Copyright (C) 2024 Custom Extension
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
YAML Configuration Loader for Calibre-Web
支持从 YAML 文件加载配置并应用到系统中
"""

import os
import sys
from typing import Optional, Dict, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from . import logger

log = logger.create()


class YamlConfigLoader:
    """YAML 配置加载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: YAML 配置文件路径，如果不指定则尝试在项目根目录查找 config.yaml
        """
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        self._loaded = False
        
        if not YAML_AVAILABLE:
            log.warning("PyYAML not installed. YAML configuration loading is disabled. "
                       "Install it with: pip install PyYAML")
            return
        
        # 确定配置文件路径
        if not self.config_path:
            # 尝试从环境变量获取
            self.config_path = os.environ.get('CALIBRE_CONFIG_FILE')
        
        if not self.config_path:
            # 默认路径：项目根目录下的 config.yaml
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
            self.config_path = os.path.join(base_dir, 'config.yaml')
        
        # 加载配置
        self.load()
    
    def load(self) -> bool:
        """
        加载 YAML 配置文件
        
        Returns:
            是否成功加载配置
        """
        if not YAML_AVAILABLE:
            return False
        
        if not os.path.exists(self.config_path):
            log.info(f"YAML config file not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f) or {}
            
            self._loaded = True
            log.info(f"Successfully loaded YAML config from: {self.config_path}")
            return True
            
        except yaml.YAMLError as e:
            log.error(f"Error parsing YAML config file: {e}")
            return False
        except Exception as e:
            log.error(f"Error loading YAML config file: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """检查配置是否已加载"""
        return self._loaded
    
    def get(self, *keys, default=None):
        """
        获取配置值（支持嵌套键）
        
        Args:
            *keys: 配置键路径，例如 get('server', 'port')
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        value = self.config_data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def apply_to_cli_params(self, cli_param):
        """
        将 YAML 配置应用到 CLI 参数对象
        
        Args:
            cli_param: CliParameter 对象
        """
        if not self.is_loaded():
            return
        
        # 服务器配置
        host = self.get('server', 'host')
        if host and not cli_param.ip_address:
            cli_param.ip_address = host
            log.info(f"Using host from YAML config: {host}")
        
        # SSL 配置
        cert_file = self.get('server', 'ssl', 'cert_file')
        key_file = self.get('server', 'ssl', 'key_file')
        if cert_file and key_file and not cli_param.certfilepath:
            cli_param.certfilepath = cert_file
            cli_param.keyfilepath = key_file
            log.info(f"Using SSL config from YAML: cert={cert_file}, key={key_file}")
        
        # 日志路径
        log_file = self.get('logging', 'file')
        if log_file and not cli_param.logpath:
            cli_param.logpath = log_file
            log.info(f"Using log file from YAML config: {log_file}")
        
        # 高级选项
        if self.get('advanced', 'allow_localhost') and not cli_param.allow_localhost:
            cli_param.allow_localhost = True
            log.info("Enabled allow_localhost from YAML config")
        
        if self.get('advanced', 'enable_reconnect') and not cli_param.reconnect_enable:
            cli_param.reconnect_enable = True
            log.info("Enabled reconnect endpoint from YAML config")
    
    def apply_to_config(self, config):
        """
        将 YAML 配置应用到数据库配置对象
        
        Args:
            config: ConfigSQL 对象
        """
        if not self.is_loaded():
            return
        
        # 服务器配置
        port = self.get('server', 'port')
        if port is not None:
            config.config_port = int(port)
            log.info(f"Using port from YAML config: {port}")
        
        external_port = self.get('server', 'external_port')
        if external_port is not None:
            config.config_external_port = int(external_port)
        
        # SSL 配置
        cert_file = self.get('server', 'ssl', 'cert_file')
        if cert_file:
            config.config_certfile = cert_file
        
        key_file = self.get('server', 'ssl', 'key_file')
        if key_file:
            config.config_keyfile = key_file
        
        # Calibre 配置
        library_path = self.get('calibre', 'library_path')
        if library_path:
            # 转换为绝对路径
            if not os.path.isabs(library_path):
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
                library_path = os.path.abspath(os.path.join(base_dir, library_path))
            config.config_calibre_dir = library_path
            log.info(f"Using Calibre library path from YAML config: {library_path}")
        
        use_gdrive = self.get('calibre', 'use_google_drive')
        if use_gdrive is not None:
            config.config_use_google_drive = bool(use_gdrive)
        
        gdrive_folder = self.get('calibre', 'google_drive_folder')
        if gdrive_folder:
            config.config_google_drive_folder = gdrive_folder
        
        # 应用配置
        title = self.get('app', 'title')
        if title:
            config.config_calibre_web_title = title
        
        books_per_page = self.get('app', 'books_per_page')
        if books_per_page is not None:
            config.config_books_per_page = int(books_per_page)
        
        random_books = self.get('app', 'random_books')
        if random_books is not None:
            config.config_random_books = int(random_books)
        
        theme = self.get('app', 'theme')
        if theme is not None:
            config.config_theme = int(theme)
        
        anon_browse = self.get('app', 'anonymous_browse')
        if anon_browse is not None:
            config.config_anonbrowse = 1 if anon_browse else 0
        
        public_reg = self.get('app', 'public_registration')
        if public_reg is not None:
            config.config_public_reg = 1 if public_reg else 0
        
        allow_upload = self.get('app', 'allow_upload')
        if allow_upload is not None:
            config.config_uploading = 1 if allow_upload else 0
        
        upload_formats = self.get('app', 'upload_formats')
        if upload_formats:
            config.config_upload_formats = upload_formats
        
        # 日志配置
        log_level_str = self.get('logging', 'level')
        if log_level_str:
            level_map = {
                'DEBUG': 10,
                'INFO': 20,
                'WARNING': 30,
                'ERROR': 40
            }
            if log_level_str.upper() in level_map:
                config.config_log_level = level_map[log_level_str.upper()]
        
        log_file = self.get('logging', 'file')
        if log_file:
            config.config_logfile = log_file
        
        access_log_enabled = self.get('logging', 'access_log', 'enabled')
        if access_log_enabled is not None:
            config.config_access_log = 1 if access_log_enabled else 0
        
        access_log_file = self.get('logging', 'access_log', 'file')
        if access_log_file:
            config.config_access_logfile = access_log_file
        
        # 邮件配置
        mail_server = self.get('mail', 'server')
        if mail_server:
            config.mail_server = mail_server
        
        mail_port = self.get('mail', 'port')
        if mail_port is not None:
            config.mail_port = int(mail_port)
        
        mail_use_ssl = self.get('mail', 'use_ssl')
        if mail_use_ssl is not None:
            config.mail_use_ssl = 1 if mail_use_ssl else 0
        
        mail_login = self.get('mail', 'login')
        if mail_login:
            config.mail_login = mail_login
        
        mail_password = self.get('mail', 'password')
        if mail_password:
            config.mail_password = mail_password
        
        mail_from = self.get('mail', 'from')
        if mail_from:
            config.mail_from = mail_from
        
        mail_size = self.get('mail', 'size_limit')
        if mail_size is not None:
            config.mail_size = int(mail_size)
        
        # 认证配置
        auth_type = self.get('auth', 'type')
        if auth_type is not None:
            config.config_login_type = int(auth_type)
        
        register_email = self.get('auth', 'register_email')
        if register_email is not None:
            config.config_register_email = bool(register_email)
        
        remote_login = self.get('auth', 'remote_login')
        if remote_login is not None:
            config.config_remote_login = bool(remote_login)
        
        session_protection = self.get('auth', 'session_protection')
        if session_protection is not None:
            config.config_session = int(session_protection)
        
        # Kobo 配置
        kobo_sync = self.get('kobo', 'sync')
        if kobo_sync is not None:
            config.config_kobo_sync = bool(kobo_sync)
        
        kobo_proxy = self.get('kobo', 'proxy')
        if kobo_proxy is not None:
            config.config_kobo_proxy = bool(kobo_proxy)
        
        # 转换器路径
        calibre_path = self.get('converters', 'calibre_path')
        if calibre_path:
            config.config_converterpath = calibre_path
        
        kepubify_path = self.get('converters', 'kepubify_path')
        if kepubify_path:
            config.config_kepubifypath = kepubify_path
        
        unrar_path = self.get('converters', 'unrar_path')
        if unrar_path:
            config.config_rarfile_location = unrar_path
        
        # 速率限制
        rate_limit_enabled = self.get('rate_limit', 'enabled')
        if rate_limit_enabled is not None:
            config.config_ratelimiter = bool(rate_limit_enabled)
        
        rate_limit_uri = self.get('rate_limit', 'storage_uri')
        if rate_limit_uri:
            config.config_limiter_uri = rate_limit_uri
        
        rate_limit_options = self.get('rate_limit', 'storage_options')
        if rate_limit_options:
            config.config_limiter_options = rate_limit_options
        
        # 高级选项
        unicode_filename = self.get('advanced', 'unicode_filename')
        if unicode_filename is not None:
            config.config_unicode_filename = bool(unicode_filename)
        
        trusted_hosts = self.get('advanced', 'trusted_hosts')
        if trusted_hosts:
            config.config_trustedhosts = trusted_hosts
        
        # Goodreads
        goodreads_enabled = self.get('services', 'goodreads', 'enabled')
        if goodreads_enabled is not None:
            config.config_use_goodreads = bool(goodreads_enabled)
        
        goodreads_key = self.get('services', 'goodreads', 'api_key')
        if goodreads_key:
            config.config_goodreads_api_key = goodreads_key
        
        # Google Books
        google_books_key = self.get('services', 'google_books', 'api_key')
        if google_books_key:
            config.config_googlebooks_api_key = google_books_key
        
        log.info("Applied YAML configuration to config object")
    
    def get_custom_config(self) -> Dict[str, Any]:
        """
        获取自定义配置部分
        
        Returns:
            自定义配置字典
        """
        return self.get('custom', default={})


# 全局配置加载器实例
_yaml_loader: Optional[YamlConfigLoader] = None


def get_yaml_loader(config_path: Optional[str] = None) -> YamlConfigLoader:
    """
    获取全局 YAML 配置加载器实例
    
    Args:
        config_path: YAML 配置文件路径
        
    Returns:
        YamlConfigLoader 实例
    """
    global _yaml_loader
    if _yaml_loader is None:
        _yaml_loader = YamlConfigLoader(config_path)
    return _yaml_loader

