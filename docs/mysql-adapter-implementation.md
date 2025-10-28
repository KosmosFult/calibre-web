# MySQL适配器实现清单

## 概述

本文档描述了将Calibre-Web从仅支持SQLite改造成支持MySQL所需的代码改动和实施方案。

**改造难度**: ⭐⭐⭐⭐ (4/5)  
**预估工作量**: 3-5个工作日  
**影响文件数**: ~20个  
**新增代码量**: ~500行  
**修改代码量**: ~700行  

---

## 核心问题分析

### 1. ATTACH DATABASE 依赖 (最高难度)

**问题**: SQLite特有的ATTACH功能用于同时访问calibre数据库和应用设置数据库

**当前代码** (db.py: 685-686):
```python
connection.execute(text("attach database '{}' as calibre;".format(dbpath)))
connection.execute(text("attach database '{}' as app_settings;".format(app_db_path)))
```

**解决方案**:
- **方案A** (推荐): 合并到单一数据库，使用prefix区分表
- **方案B**: 使用ryoung/sqlalchemy-cross-db-query实现跨数据库查询

**影响范围**: 
- `cps/db.py` 中所有setup_db相关代码
- `Data` 表的schema='calibre'定义

---

### 2. ilike() 不兼容 (中等难度)

**问题**: SQLite特有的case-insensitive LIKE

**当前代码** (多处):
```python
# db.py:928, 940, 942, 964-974
func.lower(database.name).ilike("%" + query + "%")
func.lower(Books.title).ilike("%" + title + "%")
```

**需要替换为**: 
```python
# SQLite版本
column.ilike(pattern)

# MySQL版本  
func.lower(column).like(func.lower(pattern))
# 或使用MySQL的COLLATE
column.collate('utf8mb4_unicode_ci').like(pattern)
```

**影响文件**: 
- `cps/db.py` - 约10处
- `cps/ub.py` - 需要检查是否有类似用法

---

### 3. create_function() 依赖 (中等难度)

**问题**: SQLite允许在SQL中使用Python函数

**当前代码** (db.py:1065-1067):
```python
conn.create_function("title_sort", 1, _title_sort)
conn.create_function('uuid4', 0, lambda: str(uuid4()))
conn.create_function("lower", 1, lcase)
```

**解决方案**:
- **title_sort**: 改为应用层排序或在MySQL中创建存储函数
- **uuid4**: 使用MySQL的UUID()函数或应用层生成
- **lower**: MySQL原生支持，直接使用LOWER()

**影响**: 
- 搜索和排序功能需要重构
- `get_typeahead` 等使用lower函数的查询

---

### 4. SQLite特定的配置

**当前代码**:
```python
# db.py:678-682
engine = create_adecimal('sqlite://',
                       echo=False,
                       isolation_level="SERIALIZABLE",
                       connect_args={'check_same_thread': False},
                       poolclass=StaticPool)

# db.py:684
connection.execute(text('PRAGMA cache_size = 10000;'))

# ub.py:240
__table_args__ = {'sqlite_autoincrement': True}
```

**MySQL适配**:
```python
# MySQL版本
engine = create_engine('mysql+pymysql://user:pass@host/db',
                       echo=False,
                       pool_size=10,
                       max_overflow=20,
                       pool_pre_ping=True)
```

---

### 5. Schema处理

**当前代码** (db.py:358):
```python
class Data(Base):
    __tablename__ = 'data'
    __table_args__ = {'schema': 'calibre'}
```

**说明**: 在SQLite中，schema通过ATTACH实现；在MySQL中直接使用schema参数即可。

---

## 实施方案

### 阶段1: 创建数据库适配器抽象层

#### 新建文件: `cps/db_adapter.py`

```python
from abc import ABC, abstractmethod
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Query
from enum import Enum

class DatabaseType(Enum):
    SQLITE = 'sqlite'
    MYSQL = 'mysql'
    POSTGRESQL = 'postgresql'  # 预留扩展

class DatabaseAdapter(ABC):
    """数据库抽象适配器基类"""
    
    def __init__(self, config, db_type: DatabaseType):
        self.config = config
        self.db_type = db_type
    
    @abstractmethod
    def create_engine(self, *args, **kwargs):
        """创建SQLAlchemy Engine"""
        pass
    
    @abstractmethod
    def setup_connection(self, engine, calibre_path, app_db_path):
        """设置数据库连接"""
        pass
    
    @abstractmethod
    def create_sql_functions(self, connection):
        """创建SQL函数"""
        pass
    
    @abstractmethod
    def case_insensitive_like(self, column, pattern):
        """跨数据库的case-insensitive LIKE"""
        pass
    
    @abstractmethod
    def random_function(self):
        """随机函数"""
        pass


class SQLiteAdapter(DatabaseAdapter):
    """SQLite适配器"""
    
    def __init__(self, config):
        super().__init__(config, DatabaseType.SQLITE)
    
    def create_engine(self):
        from sqlalchemy.pool import StaticPool
        return create_engine('sqlite://',
                           echo=False,
                           isolation_level="SERIALIZABLE",
                           connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    
    def setup_connection(self, engine, calibre_path, app_db_path):
        from sqlalchemy import text
        with engine.begin() as connection:
            connection.execute(text('PRAGMA cache_size = 10000;'))
            connection.execute(text("attach database '{}' as calibre;".format(calibre_path)))
            connection.execute(text("attach database '{}' as app_settings;".format(app_db_path)))
    
    def create_sql_functions(self, connection):
        from sqlite3 import OperationalError
        from uuid import uuid4
        from .db import lcase
        
        def _title_sort(title):
            # 实现title_sort逻辑
            pass
        
        try:
            if hasattr(connection, 'driver_connection'):
                conn = connection.driver_connection
            else:
                conn = connection.connection.connection
                
            conn.create_function("title_sort", 1, _title_sort)
            conn.create_function('uuid4', 0, lambda: str(uuid4()))
            conn.create_function("lower", 1, lcase)
        except OperationalError:
            pass
    
    def case_insensitive_like(self, column, pattern):
        return column.ilike(pattern)
    
    def random_function(self):
        from sqlalchemy import func
        return func.random()


class MySQLAdapter(DatabaseAdapter):
    """MySQL适配器"""
    
    def __init__(self, config):
        super().__init__(config, DatabaseType.MYSQL)
        self.calibre_schema = config.config_mysql_calibre_schema or 'calibre'
        self.app_schema = config.config_mysql_app_schema or 'calibre_web'
    
    def create_engine(self):
        db_url = config.config_mysql_url  # mysql+pymysql://user:pass@host/db
        return create_engine(db_url,
                           echo=False,
                           pool_size=10,
                           max_overflow=20,
                           pool_pre_ping=True,
                           pool_recycle=3600)
    
    def setup_connection(self, engine, calibre_path, app_db_path):
        # MySQL不需要ATTACH，直接使用schema
        # 可以选择创建schemas或使用prefix
        pass
    
    def create_sql_functions(self, connection):
        # MySQL可以创建存储函数或使用原生函数
        # UUID函数: UUID()
        # 需要在应用层实现title_sort
        pass
    
    def case_insensitive_like(self, column, pattern):
        # 方法1: 使用LOWER()
        return func.lower(column).like(func.lower(pattern))
        
        # 方法2: 使用COLLATE (需要表定义时设置)
        # return column.collate('utf8mb4_unicode_ci').like(pattern)
    
    def random_function(self):
        from sqlalchemy import func
        return func.rand()


class DatabaseAdapterFactory:
    """数据库适配器工厂"""
    
    @staticmethod
    def create_adapter(config):
        db_type = config.config_database_type or 'sqlite'
        
        if db_type == 'sqlite':
            return SQLiteAdapter(config)
        elif db_type == 'mysql':
            return MySQLAdapter(config)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
```

---

### 阶段2: 修改 CalibreDB 类

#### 修改文件: `cps/db.py`

**改动点**:

1. **导入适配器** (顶部):
```python
from .db_adapter import DatabaseAdapterFactory
```

2. **修改 CalibreDB 类** (约530行开始):
```python
class CalibreDB:
    config = None
    config_calibre_dir = None
    app_db_path = None
    adapter = None  # 新增

    @classmethod
    def update_config(cls, config, config_calibre_dir, app_db_path):
        cls.config = config
        cls.config_calibre_dir = config_calibre_dir
        cls.app_db_path = app_db_path
        # 初始化适配器
        cls.adapter = DatabaseAdapterFactory.create_adapter(config)  # 新增

    @classmethod
    def setup_db(cls, config_calibre_dir, app_db_path):
        if not config_calibre_dir:
            cls.config.invalidate()
            return None

        dbpath = os.path.join(config_calibre_dir, "metadata.db")
        if not os.path.exists(dbpath):
            cls.config.invalidate()
            return None

        try:
            # 使用适配器创建engine
            engine = cls.adapter.create_engine()  # 修改
            
            # 使用适配器设置连接
            cls.adapter.setup_connection(engine, dbpath, app_db_path)  # 修改
            
            conn = engine.connect()
        except Exception as ex:
            cls.config.invalidate(ex)
            return None

        cls.config.db_configured = True

        if not cc_classes:
            try:
                # MySQL需要指定schema
                cc = conn.execute(text("SELECT grouping_id, name, sort FROM book_groups"))
                cls.setup_db_cc_classes(cc)
            except OperationalError as e:
                log.error_or_exception(e)
                return None

        return scoped_session(sessionmaker(autocommit=False,
                                          autoflush=False,
                                          bind=engine, future=True))
```

3. **修改搜索相关方法** (约924-975行):
```python
def get_typeahead(self, database, query, replace=('', ''), tag_filter=true()):
    query = query or ''
    self.create_functions()
    # 使用适配器的case_insensitive_like
    entries = self.session.queryниdatabase).filter(tag_filter). \
        filter(self.adapter.case_insensitive_like(
            database.name, "%" + query + "%")).all()
    # ...

def search_query(self, term, config, *join):
    # ... 类似修改
```

4. **修改 create_functions 方法** (约1046行):
```python
def create_functions(self, config=None):
    try:
        conn = self.session.connection()
        # 使用适配器创建函数
        self.adapter.create_sql_functions(conn)
    except Exception as e:
        log.error_or_exception(e)
```

---

### 阶段3: 修改 Data 表定义

#### 修改文件: `cps/db.py`

**改动** (约356行):
```python
class Data(Base):
    __tablename__ = 'data'
    # 根据适配器决定是否使用schema
    if CalibreDB.adapter and CalibreDB.adapter.db_type != DatabaseType.SQLITE:
        __table_args__ = {'schema': CalibreDB.adapter.calibre_schema}
    else:
        __table_args__ = {}
```

---

### 阶段4: 修改用户数据库

#### 修改文件: `cps/ub.py`

**主要改动**:

1. **初始化引擎** (约680行):
```python
def init_db_thread():
    global Session
    engine = None
    if config.config_database_type == 'mysql':
        # MySQL配置
        db_url = config.config_mysql_url
        engine = create_engine(db_url, echo=False, pool_pre_ping=True)
    else:
        # SQLite配置
        engine = create_engine('sqlite:///{0}'.format(app_DB_path), echo=False)
    # ...
```

2. **移除 sqlite_autoincrement** (约240行):
```python
class BookRead(ub.Base):
    __tablename__ = 'bookread'
    # __table_args__ = {'sqlite_autoincrement': True}  # 删除此行
    id = Column(Integer, primary_key=True, autoincrement=True)
```

---

### 阶段5: 配置管理

#### 修改文件: `cps/config_sql.py`

**新增配置项**:
```python
class ConfigSQL(object):
    def __init__(self):
        # ... 现有配置
        self.config_database_type = 'sqlite'  # 'sqlite' 或 'mysql'
        self.config_mysql_url = ''  # mysql+pymysql://user:pass@host/db
        self.config_mysql_calibre_schema = 'calibre'
        self.config_mysql_app_schema = 'calibre_web'
```

#### 修改文件: `cps/templates/config_db.html`

**新增配置界面** (需要添加MySQL配置表单)

---

### 阶段6: 依赖管理

#### 修改文件: `requirements.txt`

**新增**:
```
# MySQL支持
pymysql>=1.0.0
cryptography>=3.0.0
```

---

## 详细改动清单

### 高优先级改动

| 文件 | 行号范围 | 改动类型 | 说明 | 难度 |
|------|---------|---------|------|------|
| `cps/db.py` | 625-706 | 重构 | setup_db方法，使用适配器 | ⭐⭐⭐⭐⭐ |
| `cps/db.py` | 1046-1069 | 重构 | create_functions，依赖适配器 | ⭐⭐⭐⭐ |
| `cps/db.py` | 356-358 | 修改 | Data表schema定义 | ⭐⭐ |
| `cps/db.py` | 924-975 | 批量修改 | 所有ilike替换 | ⭐⭐⭐ |
| `cps/db_adapter.py` | 新建 | 新增 | 适配器抽象层 | ⭐⭐⭐⭐ |

### 中优先级改动

| 文件 | 行号范围 | 改动类型 | 说明 | 难度 |
|------|---------|---------|------|------|
| `cps/ub.py` | 680-693 | 重构 | init_db初始化引擎 | ⭐⭐⭐ |
| `cps/ub.py` | 240 | 删除 | sqlite_autoincrement | ⭐ |
| `cps/ub.py` | 734 | 重构 | get_new_session_instance | ⭐⭐ |
| `cps/config_sql.py` | 多处 | 新增 | MySQL配置项 | ⭐⭐ |

### 低优先级改动

| 文件 | 改动类型 | 说明 | 难度 |
|------|---------|------|------|
| `requirements.txt` | 新增 | 添加pymysql依赖 | ⭐ |
| `cps/templates/config_db.html` | 修改 | 添加MySQL配置UI | ⭐⭐ |
| `cps/__init__.py` | 修改 | 初始化适配器 | ⭐ |
| `cps/about.py` | 修改 | 显示MySQL版本信息 | ⭐ |

---

## 数据库迁移方案

### 从SQLite迁移到MySQL

1. **导出SQLite数据**:
```bash
sqlite3 metadata.db .dump > sqlite_dump.sql
```

2. **转换SQL语法**:
   - 移除ATTACH语句
   - 移除sqlite_autoincrement
   - 转换数据类型
   - 调整索引定义

3. **导入MySQL**:
```bash
mysql -u user -p database < converted_dump.sql
```

4. **设置字符集**:
```sql
ALTER DATABASE database CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

---

## 测试清单

### 功能测试

- [ ] 数据库连接初始化
- [ ] 书籍列表加载
- [ ] 搜索功能（区分大小写）
- [ ] 作者/标签/系列浏览
- [ ] 电子书详情查看
- [ ] 用户登录/认证
- [ ] 书籍上传
- [ ] 元数据编辑
- [ ] 书架管理

### 性能测试

- [ ] 大数据库（>10万本书）加载时间
- [ ] 搜索响应时间
- [ ] 并发用户测试
- [ ] 内存使用情况

### 兼容性测试

- [ ] MySQL 5.7
- [ ] MySQL 8.0
- [ ] MariaDB 10.x

---

## 风险和注意事项

### 已知风险

1. **数据迁移风险**: 
   - SQLite和MySQL在数据类型、约束上有差异
   - 需要充分的测试和备份

2. **性能风险**:
   - MySQL的网络延迟可能影响响应速度
   - 需要合理配置连接池

3. **自定义函数风险**:
   - title_sort函数在MySQL中需要重实现
   - 可能影响某些排序功能

### 注意事项

1. **备份**: 迁移前必须完整备份SQLite数据库
2. **字符集**: 确保使用utf8mb4支持emoji等字符
3. **索引**: MySQL和SQLite的索引策略不同，需要验证
4. **事务**: MySQL默认autocommit行为不同

---

## 后续扩展

### 可能的扩展

- PostgreSQL支持
- SQL Server支持
- 连接池监控
- 读写分离
- 分库分表（大数据量场景）

---

## 参考资料

- [SQLAlchemy Dialects](https://docs.sqlalchemy.org/en/14/dialects/)
- [SQLite to MySQL Migration](https://dev.mysql.com/doc/refman/8.0/en/sqlite-to-mysql.html)
- [PyMySQL Documentation](https://pymysql.readthedocs.io/)

---

**文档版本**: 1.0  
**最后更新**: 2024-12-19  
**作者**: AI Assistant

