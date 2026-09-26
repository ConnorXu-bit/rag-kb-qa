# HTTP 接口

## GET /health

健康检查，返回 status 和 chunks 两个字段，chunks 是当前库存的片段数。
部署后用来确认服务活着、知识库有没有内容。

## POST /ingest

入库一篇文档，请求体是 source 和 text 两个字段。
同一 source 重复导入会先删掉旧片段再写入，保证幂等，不会越导越多。
返回里带上这次切出来的片段数。

## POST /query

提问，请求体只有一个 question 字段。
返回 answer、used_llm 和 sources 三部分。
used_llm 是布尔值：命中为空时它是 false，说明这次没有调用模型，走的是兜底话术。

## 参数校验

source、text、question 都要求非空，长度不足会被 pydantic 拦下并返回 422。
校验放在接口层，业务层就不需要重复写防御代码。
