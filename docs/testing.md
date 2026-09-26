# 测试策略

## 依赖注入与测试替身

Embedder、LLM、VectorStore 都是构造器注入，测试里换成假实现：
FakeEmbedder 给确定性向量，FakeLLM 按剧本返回答案，VectorStore 用内存 SQLite。
接口层用 create_app(service) 注入，业务代码一行不改，测的是真实代码路径。

## FakeEmbedder 的边界

FakeEmbedder 是词袋哈希：同一段文本永远得到同一个向量，共享词汇越多余弦相似度越高。
它能验证流程和排序，但验证不了真实语义召回：
两组意思相同、用词完全不同的句子，在词袋模型里相似度可能是 0。
所以线上要换成 OpenAIEmbedder，评估语义召回也必须用真实向量。

## 覆盖了哪些边界

测试覆盖切分边界与参数校验、向量维度校验、RRF 融合去重、无命中兜底，以及 HTTP 参数校验。
跑起来不联网、不花钱，CI 在 Python 3.10 / 3.12 / 3.14 上运行。
