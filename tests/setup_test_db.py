"""
setup_test_db.py —— 创建/重置测试专用数据库

职责：
    在 hadoop100 上重建 ai_customer_service_test 库：建 9 张表 + 从开发库
    （ai_customer_service）整表拷贝种子数据（含 335 条向量），并把每个
    自增序列推到正确位置。

为什么需要它（2026-11-12）：
    此前测试直接跑在开发库上。测试虽然用 SAVEPOINT 回滚、不留数据，
    但它**读**库里已提交的状态 —— 用例写着"李四还没有会话""张三会话只有
    4 条种子消息"，你一用系统聊天，这些前提就破了，测试变红。
    根治办法：测试跑在独立克隆库上，开发库随便用，互不影响。

用法（Windows，项目根目录）：
    D:\\app\\anaconda\\envs\\llm1\\python.exe tests\\setup_test_db.py

什么时候需要重跑：
    - 想让测试用上新的种子数据时（比如给开发库加了新订单）
    - 测试库被误删时
    测试期间的写入都会被 SAVEPOINT 回滚掉，所以正常情况**永远不用重跑**。

不负责：
    修改开发库（只读拷贝源）、在线服务的连接（服务连的是开发库）。
"""

import sys

import psycopg

# ---------- 连接参数（与 .env 的开发库保持同一账号，只是库名不同） ----------
HOST = "192.168.218.100"
PORT = 5432
USER = "ai_cs"
PASSWORD = "ai_cs_pwd_2026"
DEV_DB = "ai_customer_service"
TEST_DB = "ai_customer_service_test"

# 9 张表的建表 SQL —— 按**开发库的真实结构**生成（2026-11-12 用
# information_schema 逐列核对过），不是照抄设计文档：文档附录漏了
# users.role、products.status 默认值也和实际不符（'on_sale' 而非 '在售'）。
# 列顺序必须与开发库完全一致：COPY (FORMAT binary) 按位置对齐，
# 列序错位会报 "row field count is N, expected M" 或静默串列。
# 表间有外键依赖，建表顺序不能乱：先父表后子表。
DDL = """
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL   PRIMARY KEY,
    username      VARCHAR(50) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    nickname      VARCHAR(50),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    role          VARCHAR(20) NOT NULL DEFAULT 'customer',
    CONSTRAINT uq_users_username UNIQUE (username)
);
CREATE TABLE IF NOT EXISTS conversations (
    id             BIGSERIAL  PRIMARY KEY,
    user_id        BIGINT     NOT NULL REFERENCES users(id),
    status         VARCHAR(20) NOT NULL DEFAULT 'AI',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_by     BIGINT     REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);
CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL  PRIMARY KEY,
    conversation_id BIGINT     NOT NULL REFERENCES conversations(id),
    role            VARCHAR(10) NOT NULL,
    content         TEXT       NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS products (
    id          BIGSERIAL PRIMARY KEY,
    name        VARCHAR(200) NOT NULL,
    category    VARCHAR(50),
    price       NUMERIC(10,2) NOT NULL,
    stock       INT           NOT NULL DEFAULT 0,
    status       VARCHAR(20)   NOT NULL DEFAULT 'on_sale',
    description TEXT,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS orders (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT NOT NULL REFERENCES users(id),
    status       VARCHAR(20) NOT NULL,
    total_amount NUMERIC(10,2) NOT NULL,
    address      VARCHAR(500),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);
CREATE TABLE IF NOT EXISTS order_items (
    id           BIGSERIAL PRIMARY KEY,
    order_id     BIGINT NOT NULL REFERENCES orders(id),
    product_id   BIGINT NOT NULL REFERENCES products(id),
    product_name VARCHAR(200) NOT NULL,
    quantity     INT NOT NULL,
    unit_price   NUMERIC(10,2) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id         BIGSERIAL PRIMARY KEY,
    title      VARCHAR(200) NOT NULL,
    file_name  VARCHAR(255),
    status     VARCHAR(20) NOT NULL DEFAULT 'processing',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS document_chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES knowledge_documents(id),
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1024),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS agent_runs (
    id                 BIGSERIAL PRIMARY KEY,
    conversation_id    BIGINT,
    user_id            BIGINT NOT NULL,
    question           TEXT NOT NULL,
    reply_type         VARCHAR(32),
    validation_status  VARCHAR(32) NOT NULL,
    error_codes        TEXT[] NOT NULL DEFAULT '{}',
    tool_call_count    INT NOT NULL DEFAULT 0,
    duration_ms        INT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation ON agent_runs (conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs (validation_status);
"""

# 拷贝顺序 = 外键依赖顺序（父表在前），与 DDL 一致
TABLES = (
    "users",
    "products",
    "conversations",
    "messages",
    "orders",
    "order_items",
    "knowledge_documents",
    "document_chunks",
    "agent_runs",
)


def recreate_test_db() -> None:
    """
    删除并重建空的测试库（含 vector 扩展）。

    异常：
        psycopg.OperationalError：账号没有 CREATEDB 权限或服务器不可达。
        本脚本是命令行工具，直接让异常冒出去给完整的报错信息。
    """
    # DROP/CREATE DATABASE 不能在事务里跑，必须 autocommit
    with psycopg.connect(host=HOST, port=PORT, dbname="postgres",
                         user=USER, password=PASSWORD, autocommit=True) as conn:
        # 有人/有测试正连着测试库时 DROP 会失败，提示信息里直接说明原因
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        conn.execute(f"CREATE DATABASE {TEST_DB}")
    with psycopg.connect(host=HOST, port=PORT, dbname=TEST_DB,
                         user=USER, password=PASSWORD, autocommit=True) as conn:
        # 难点：vector 扩展是"库级别"的，新库不会自动继承开发库的扩展，
        # 必须先建扩展再建 document_chunks（vector 列类型依赖它）。
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(DDL)


def copy_data() -> None:
    """
    把开发库 9 张表的数据整表拷贝进测试库，并把自增序列推到 max(id)。

    实现方式：psycopg3 的 COPY TO STDOUT / COPY FROM STDIN（文本协议）。
    开发库和测试库在同一台服务器上，数据在本机内存里过一道手就完成，
    335 条 1024 维向量也只有几 MB，没有必要走服务器端 pg_dump。

    # 难点：为什么拷完必须 setval？
    # 显式插入了带 id 的行，但序列本身还停在 1。测试用例会建新会话/新消息，
    # INSERT 不带 id 时会从 1 开始自增，直接撞主键冲突。
    # 所以每张表都要把序列推到 max(id)，等价于 seed 脚本末尾的 setval。
    """
    with psycopg.connect(host=HOST, port=PORT, dbname=DEV_DB,
                         user=USER, password=PASSWORD) as src, \
         psycopg.connect(host=HOST, port=PORT, dbname=TEST_DB,
                         user=USER, password=PASSWORD) as dst:
        dst.autocommit = True
        for table in TABLES:
            with src.cursor() as out, dst.cursor() as inp:
                # 难点：两端的 COPY 格式必须一致。
                # psycopg3 的 copy 对象读写的是原始字节，格式由 SQL 决定 ——
                # TO STDOUT 不声明时驱动默认二进制，FROM STDIN 不声明时按文本解析，
                # 一边二进制一边文本就会报 "extra data after last expected column"。
                # 两边都显式写 (FORMAT binary)，字节原样过手，向量列也不会丢精度。
                with out.copy(f"COPY {table} TO STDOUT (FORMAT binary)") as cp:
                    # 难点：read() 一次只返回一段数据，必须循环读到空串。
                    # 只读一次就退出 with 块，COPY 命令没收尾，连接停留在
                    # "命令进行中"状态，下一张表的 COPY 会报
                    # "another command is already in progress"。
                    rows = bytearray()
                    while chunk := cp.read():
                        rows += chunk
                with inp.copy(f"COPY {table} FROM STDIN (FORMAT binary)") as cp:
                    cp.write(rows)
                # 空表没有 max(id)，跳过 setval（序列保持初始值 1 正确）
                max_id = inp.execute(
                    f"SELECT COALESCE(max(id), 0) FROM {table}"
                ).fetchone()[0]
                if max_id:
                    inp.execute(
                        f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), %s)",
                        (max_id,),
                    )
            print(f"  {table:<20} 拷贝完成，序列推进到 {max_id}")


def main() -> int:
    """命令行入口：重建 → 拷贝 → 汇总。返回 0 成功。"""
    print(f"重建测试库 {TEST_DB} …")
    recreate_test_db()
    print("拷贝开发库种子数据 …")
    copy_data()

    # 自检：测试库应能查到种子数据
    with psycopg.connect(host=HOST, port=PORT, dbname=TEST_DB,
                         user=USER, password=PASSWORD) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM document_chunks WHERE embedding IS NOT NULL")
            vectors = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM conversations")
            convs = cur.fetchone()[0]
    print(f"完成：测试库 {TEST_DB} 就绪（{convs} 个会话 / {vectors} 条带向量切块）。")
    print("现在可以正常运行 pytest，它只会碰这个库，不再影响开发库。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
