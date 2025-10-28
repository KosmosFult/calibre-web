# 适配器使用示例详解

## 核心思路

**适配器不用替换所有的数据库查询**，它只需要：
1. 在**初始化时**做数据库特定的设置
2. 提供**查询构造器**来处理差异

---

## 具体改造示例

### 示例1: 初始化阶段使用适配器

#### 改造前 (db.py: 665-706)

```python
@classmethod
def setup_db(cls, config_calibre_dir, app_db_path):
    dbpath = os.path.join(config_calibre_dir, "metadata.db")
    
    # SQLite特定的初始化
    engine = create_engine('sqlite://', ...)
    with engine.begin() as connection:
        connection.execute(text('PRAGMA cache_size = 10000;'))
        connection.execute(text("attach database '{}' as calibre;".format(dbpath)))
        connection.execute(text("attach database '{}' as app_settings;".format(app_db_path)))
    
    conn = engine.connect()
    return scoped_session(sessionmaker(bind=engine))
```

#### 改造后 (使用适配器)

```python
@classmethod
def setup_db(cls, config_calibre_dir, app_db_path):
    dbpath = os.path.join(config_calibre_dir, "metadata.db")
    
    # 1. 根据配置创建engine
    if cls.config.config_database_type == 'mysql':
        engine = create_engine(
            cls.config.config_mysql_url,
            echo=False,
            pool_pre_ping=True
        )
    else:  # SQLite
        engine = create_engine('sqlite://', ...)
    
    # 2. 使用适配器做数据库特定的设置
    adapter = cls.get_adapter()
    adapter.setup_connection(engine, dbpath, app_db_path)
    
    # 3. 保存适配器供后续使用
    cls.adapter = adapter
    
    conn = engine.connect()
    return scoped_session(sessionmaker(bind=engine))

@classmethod
def get_adapter(cls):
    """获取数据库适配器"""
    if not hasattr(cls, '_adapter') or cls._adapter is None:
        db_type = cls.config.config_database_type or 'sqlite'
        cls._adapter = DatabaseAdapter(db_type)
    return cls._adapter
```

**关键点**: 适配器在这里只负责**连接初始化**，不改变查询逻辑！

---

### 示例2: 搜索查询改造

#### 改造前 (db.py: 928-929)

```python
def get_typeahead(self, database, query, replace=('', ''), tag_filter=true()):
    query = query or ''
    self.create_functions()
    
    # 直接使用 SQLite 的 ilike
    entries = self.session.query(database).filter(tag_filter). \
        filter(func.lower(database.name).ilike("%" + query + "%")).all()
    
    return json.dumps([dict(name=r.name.replace(*replace)) for r in entries])
```

#### 改造后 (使用适配器)

```python
def get_typeahead(self, database, query, replace=('', ''), tag_filter=true()):
    query = query or ''
    self.create_functions()
    
    # 使用适配器的方法来构造跨数据库的查询
    like_expr = self.__class__.adapter.case_insensitive_like(
        database.name, 
        "%" + query + "%"
    )
    
    entries = self.session.query(database).filter(tag_filter). \
        filter(like_expr).all()
    
    return json.dumps([dict(name=r.name.replace(*replace)) for r in entries])
```

**实际执行效果**:
- **SQLite**: `like_expr` = `func.lower(database.name).ilike("%query%")`
- **MySQL**: `like_expr` = `func.lower(database.name).like(func.lower("%query%"))`

---

### 示例3: 复杂的搜索查询改造

#### 改造前 (db.py: 964-968)

```python
filter_expression = [
    Books.tags.any(func.lower(Tags.name).ilike("%" + term + "%")),
    Books.series.any(func.lower(Series.name).ilike("%" + term + "%")),
    Books.authors.any(and_(*q)),
    Books.publishers.any(func.lower(Publishers.name).ilike("%" + term + "%")),
    func.lower(Books.title).ilike("%" + term + "%")
]
```

#### 改造后 (使用适配器)

```python
# 定义辅助方法，方便复用
def _build_case_insensitive_filter(self, column, term):
    """构建跨数据库的不区分大小写过滤"""
    return self.__class__.adapter.case_insensitive_like(column, "%" + term + "%")

# 然后在查询中使用
filter_expression = [
    Books.tags.any(self._build_case_insensitive_filter(Tags.name, term)),
    Books.series.any(self._build_case_insensitive_filter(Series.name, term)),
    Books.authors.any(and_(*q)),
    Books.publishers.any(self._build_case_insensitive_filter(Publishers.name, term)),
    self._build_case_insensitive_filter(Books.title, term)
]
```

---

## 完整的适配器实现示例

### cps/db_adapter.py

```python
from sqlalchemy import func, text

class DatabaseAdapter:
    """数据库适配器 - 处理不同数据库的差异"""
    
    def __init__(self, db_type):
        self.db_type = db_type
    
    def setup_connection(self, engine, metadata_path, app_path):
        """
        数据库特定的连接设置
        只在初始化时调用一次
        """
        if self.db_type == 'sqlite':
            # SQLite需要ATTACH
            with engine.begin() as conn:
                conn.execute(text('PRAGMA cache_size = 10000;'))
                conn.execute(text(f"ATTACH DATABASE '{metadata_path}' AS calibre"))
                conn.execute(text(f"ATTACH DATABASE '{app_path}' AS app_settings"))
        elif self.db_type == 'mysql':
            # MySQL不需要任何特殊操作
            # 直接连接就行，schema在URL中指定
            pass
    
    def case_insensitive_like(self, column, pattern):
        """
        返回一个SQLAlchemy表达式，用于不区分大小写的LIKE查询
        
        参数:
            column: SQLAlchemy的Column对象
            pattern: 搜索模式字符串 (带%)
        
        返回:
            SQLAlchemy表达式
        """
        if self.db_type == 'sqlite':
            # SQLite原生支持ilike
            return column.ilike(pattern)
        else:
            # MySQL用LOWER()函数
            return func.lower(column).like(func.lower(pattern))
    
    def random_function(self):
        """返回随机函数的表达式"""
        if self.db_type == 'sqlite':
            return func.random()
        else:
            return func.rand()
    
    def create_sql_functions(self, connection):
        """
        创建数据库特定的SQL函数
        只在SQLite中需要，MySQL使用原生函数
        """
        if self.db_type == 'sqlite':
            from sqlite3 import OperationalError
            try:
                # 获取底层的sqlite connection
                if hasattr(connection, 'driver_connection'):
                    conn = connection.driver_connection
                else:
                    conn = connection.connection.connection
                
                # 定义title_sort函数
                def _title_sort(title):
                    from .db import strip_whitespaces
                    title_pat = re.compile(self.config.config_title_regex, re.IGNORECASE)
                    match = title_pat.search(title)
                    if match:
                        prep = match.group(1)
                        title = title[len(prep):] + ', ' + prep
                    return strip_whitespaces(title)
                
                # 定义lower函数
                def lcase(s):
                    try:
                        return unidecode.unidecode(s.lower())
                    except Exception:
                        return s.lower()
                
                # 注册函数
                conn.create_function("title_sort", 1, _title_sort)
                conn.create_function("lower", 1, lcase)
            except OperationalError:
                pass
        # MySQL不需要，使用原生函数
```

---

## 实际使用流程图

```
启动应用
    ↓
读取配置 (config_database_type = 'mysql' or 'sqlite')
    ↓
创建适配器 (DatabaseAdapter)
    ↓
创建Engine (使用不同的URL)
    ↓
调用 adapter.setup_connection() 
    ↓
    ├─ SQLite: ATTACH databases
    └─ MySQL: 什么都不做
    ↓
保存适配器到 CalibreDB.adapter
    ↓
后续查询时...
    ↓
构造查询表达式
    ↓
    ├─ 需要ilike → 调用 adapter.case_insensitive_like()
    ├─ 需要随机 → 调用 adapter.random_function()
    └─ 其他查询 → 正常使用 (无需改动)
    ↓
执行查询 (session.query(...))
    ↓
返回结果
```

---

## 具体改动位置汇总

### 需要改动的地方 (共约15处)

#### 1. db.py - setup_db() 方法 (~30行)
```python
# 第665-706行
@classmethod
def setup_db(cls, config_calibre_dir, app_db_path):
    # 添加：根据配置选择数据库
    # 添加：调用适配器初始化
```

#### 2. db.py - get_typeahead() 方法 (~3行)
```python
# 第928-929行
# 原来: filter(func.lower(database.name).ilike(...))
# 改为: filter(adapter.case_insensitive_like(database.name, ...))
```

#### 3. db.py - search_query() 方法 (~10行)
```python
# 第964-974行
# 所有ilike替换为 adapter.case_insensitive_like()
```

#### 4. db.py - create_functions() 方法 (~10行)
```python
# 第1046-1069行
# 改为调用 adapter.create_sql_functions()
```

#### 5. config_sql.py - 配置类 (~5行)
```python
# 添加配置项:
# config_database_type
# config_mysql_url
```

### 不需要改动的地方 (95%的代码)

所有**常规查询**都不用改：

```python
# 这些都不需要改！
self.session.query(Books).filter(Books.id == book_id).first()
self.session.query(Authors).filter(Authors.name == name).all()
session.query(Tags).count()
```

**为什么不用改？**
- SQLAlchemy已经处理了不同数据库的SQL方言差异
- 只有极少数SQLite特有的功能需要适配器

---

## 完整的调用示例

### 在 Flask 路由中使用

```python
# 在某个视图函数中
@bp.route('/search')
def search():
    # 获取数据库实例 (已经有适配器)
    db = calibre_db
    
    # 执行搜索 (不需要任何改动！)
    term = request.args.get('q', '')
    results = db.search_query(term, config)
    
    # search_query内部使用了适配器的case_insensitive_like
    # 但调用者完全不需要知道！
    return render_template('results.html', results=results)
```

### 适配器如何被调用

```python
# 第一次初始化时
CalibreDB.setup_db(...)
    ↓
adapter = DatabaseAdapter('mysql')
    ↓
adapter.setup_connection(engine, ...)
    ↓
CalibreDB.adapter = adapter  # 保存起来

# 后续查询时
search_query(term, config)
    ↓
filter(Books.title.ilike("%term%"))  # 不对！
    ↓
# 改为:
like_expr = CalibreDB.adapter.case_insensitive_like(Books.title, "%term%")
filter(like_expr)
    ↓
# MySQL执行: WHERE LOWER(books.title) LIKE LOWER('%term%')
# SQLite执行: WHERE books.title ILIKE '%term%'
```

---

## 总结

### 适配器的本质

**适配器不是替换查询**，而是：
1. ✅ **在初始化时**做数据库特定设置
2. ✅ **提供查询构造方法**来处理差异
3. ❌ **不改变现有的查询调用方式**

### 改动原则

- **95%的代码不用改**: `session.query()`, `.filter()`, `.first()` 等都照常使用
- **只改5%**: 涉及 `ilike`、`ATTACH`、`create_function` 的特殊情况
- **对调用者透明**: 上层代码完全不需要知道用的什么数据库

### 具体数字

- **总改动行数**: ~80行 (在1100行的db.py中)
- **新增代码**: ~100行 (适配器类)
- **影响方法数**: 4个方法 (setup_db, get_typeahead, search_query, create_functions)
- **工作量**: 约2-3小时

---

**关键理解**: 适配器是**查询构造的辅助工具**，不是查询的执行器。大部分代码都不需要改！

