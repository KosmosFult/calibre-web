import redis
from redis.exceptions import RedisError
from sqlalchemy import func
from . import ub
from flask import current_app
import time

class CalibreRedis:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.StrictRedis(
            host=host, 
            port=port, 
            db=db, 
            decode_responses=True,
            socket_connect_timeout=3
        )
        self.config = None
        self.calibre_db = None
        self.db = None
        self.inited = False
        self.sync_interval = 3600 * 24

    def init_redis_data(self, config, calibre_db, db, app):
        if(self.inited):
            return
        """初始化或同步 Redis 热门书籍数据"""
        try:
            # 如果 key 不存在，从数据库加载数据
            with app.app_context():
                self.config = config
                self.db = db
                self.calibre_db = calibre_db
                # if not self.client.exists("hot_books:downloads"):
                #     self.sync_from_db()
                self.inited = True
        except RedisError as e:
            print(f"Redis 初始化失败: {e}")
            
    def sync_from_db(self):
            
        ranking_factor = 1.5
        ranking_num = int(self.config.config_books_per_page * ranking_factor)

        try:
            # 查询书籍下载量
            download_stats = ub.session.query(
                ub.Downloads.book_id,
                func.count(ub.Downloads.book_id).label('download_count')
            ).group_by(ub.Downloads.book_id)\
             .order_by(func.count(ub.Downloads.book_id).desc())\
             .limit(ranking_num)\
             .all()

            # 缓存到Redis
            with self.client.pipeline() as pipe:
                pipe.delete("hot_books:ranking")
                # 存储每本书的下载量
                for book_id, count in download_stats:
                    pipe.zadd("hot_books:ranking", {book_id: count})
                
                # # 设置过期时间（可选）
                # pipe.expire("books:download_ranking", 86400)  # 24小时
                pipe.execute()

            print(f"Redis 数据已同步（共 {len(download_stats)} 本书）")

        except Exception as e:
            print(f"数据库同步失败: {e}")
            raise  # 重新抛出异常以便追踪
        
    def get_top_books(self, limit=10):
        """获取热门书籍排行榜"""
        return [
            (int(bid), int(count)) 
            for bid, count in 
            self.client.zrevrange("hot_books:ranking", 0, limit-1, withscores=True)
        ]


    def increment_download(self, book_id):
        self.client.zincrby("hot_books:ranking", 1, book_id)
        
    def start_sync_task(self, app):
        """启动定时同步任务"""
        import threading
        def task():
            while True:
                with app.app_context():
                    if self.inited:
                        self.sync_from_db()
                time.sleep(self.sync_interval)
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()
            
