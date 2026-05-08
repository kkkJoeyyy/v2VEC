"""
FastAPI 应用入口
═══════════════════════════════════════════════════════
职责：
  - 创建 FastAPI 实例（通过工厂函数 create_app）
  - 挂载静态文件服务（前端 SPA）
  - 注册 API 路由（POST /upload）
  - 提供根路径页面（index.html）

启动方式：
    uvicorn src.server:app --host 0.0.0.0 --port 8000

工厂函数设计：
  create_app() 返回 FastAPI 实例，便于测试时隔离创建应用。
  测试代码可以 import create_app 来获取干净的 app 实例。
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import STATIC_DIR
from src.routes.upload_routes import router as upload_router


def create_app() -> FastAPI:
    """构建 FastAPI 应用实例。

    返回已配置完所有路由和中间件的 app，可直接用于：
      - uvicorn.run() 生产启动
      - TestClient() 单元测试
    """
    app = FastAPI(title="V2VEC — Interview Speech to Q&A")

    # 挂载静态文件目录
    # /static/script.js → app/static/script.js
    # /static/favicon.ico → app/static/favicon.ico
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # 注册 /upload API 路由
    app.include_router(upload_router)

    # 根路径 → 返回前端单页面
    @app.get("/")
    async def index():
        """返回前端 SPA 页面（index.html）。"""
        index_path = f"{STATIC_DIR}/index.html"
        with open(index_path, encoding="utf-8") as f:
            return HTMLResponse(f.read())

    return app


# 模块级 app 实例
# 这是 uvicorn 默认寻找的入口（uvicorn src.server:app）
app = create_app()
