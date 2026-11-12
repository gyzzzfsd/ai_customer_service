# 商城 · 智能客服

登录 + 多轮聊天 + 工具调用 + harness 五关卡校验 + agent_runs 观测 + RAG 知识库检索 + 取消订单落库 + 转人工 + WebSocket 推送 + 前端页面。
数据库hadoop100（PostgreSQL + pgvector），大模型用智谱 GLM，embedding 用本地 bge-m3（GPU）。



---

## 登录账号（种子数据）

| 用户名    | 密码    | 昵称   | 订单                              |

| zhangsan | 123456 | 张三    | 3 单（订单1 待发货 / 订单2 已完成 / 订单3 已取消） |
| lisi     | 123456 | 李四    | 1 单（订单4，用于演示数据隔离） <br/>         |
| wangwu   | 123456 | 王五    | ..........                               |
| agent001 | 123456 | 客服小王 | 人工坐席                                 |

---

## 启动前端

先启动后端（PyCharm 运行 run.py），再开一个命令行窗口起静态服务器：

```powershell
cd D:\allCode\AI_Model\project\AI_customer_service
D:\app\anaconda\envs\llm1\python.exe -m http.server 8080 --directory web
```

浏览器打开 <http://127.0.0.1:8080/login.html>。

页面说明（两个文件，无 node、无构建）：

- `login.html` —— 登录页。顾客用 zhangsan / lisi，坐席用 agent001。
- `chat.html` —— 一个页面两副面孔，按登录角色自动切换：
  - **顾客视图**：聊天气泡；AI 建议"取消订单"时出现红色确认按钮（点击后真实落库）；
    AI 建议转人工或点【转接人工客服】后，状态横幅提示"转接中/人工服务中"，
    人工回复经 WebSocket 实时到达（WS 断线自动降级为 3 秒轮询）。
  - **坐席视图**：左侧待接管队列（含用户昵称和最近消息预览），
    点【接管】后右侧展开完整聊天记录，可回复、可"结束并转回 AI / 关闭会话"。

## 目录说明

```
app/            后端全部代码，一个功能一个文件，平铺不建子目录
  main.py       启动装配、挂路由、健康检查
  config.py     读 .env
  db.py         连接池（不含任何业务 SQL）
  security.py   密码哈希、JWT、当前用户依赖
  llm.py        调智谱、重试、模型降级
  prompts.py    提示词与兜底话术（改话术只动这一个文件）
  embedding.py  bge-m3 懒加载单例 + encode()（归一化 1024 维）
  rag.py        知识库向量检索（<=> 余弦 + 阈值过滤）
  ingest.py     命令行回填 document_chunks.embedding（可重复跑）
  tools.py      3 个只读工具的 SQL + schema + 分发
  agent.py      一次问答的编排（工具循环）
  refine.py     收口：结构化 → 五关卡校验 → 补查 → 重写 → 兜底
  json_repair.py 模型输出 JSON 的容错解析（代码块/废话/裸引号）
  harness.py    五关卡校验（V1~V4 判断 + 证据集合，纯函数）
  run_log.py    agent_runs 观测表写入 + 校验摘要
  schemas.py    AgentOutput、ToolResult、接口 DTO
  api_auth.py   登录
  api_chat.py   会话、发消息（人工模式分流）
  api_actions.py 取消订单确认（UPDATE 三重条件）
  api_handoff.py 转人工 + 坐席端接口（队列/接管/回复/结束）
  ws.py         WebSocket 连接管理与实时推送
  log_setup.py  日志配置（唯一初始化日志的地方）
web/            前端（两个页面，纯原生 HTML/JS）
  login.html    登录页
  chat.html     顾客聊天 + 坐席工作台（按角色切换）
run.py          PyCharm 启动入口
tests/          pytest 用例
logs/app.log    运行日志（5MB 轮转，留 3 份）
script/         设计文档与代码规范
document/       知识库手册原文与切块脚本
```

