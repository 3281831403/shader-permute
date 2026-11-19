# shader-permute

## 题面

着色器变体一多，首次进入场景就卡死：键名大小写没有归一，`USE_FOG` 与 `use_fog` 被当成两个变体；布尔值 `true`/`"1"`/`1` 没有统一，同一个开关出现三种键；变体展开顺序依赖 set 迭代，两次运行结果不同；展开不设上限，2^20 个组合直接把内存打满；缓存键只包含源码哈希，pipeline 版本变了仍然命中旧变体；缓存键对 dict 插入顺序敏感，同一份配置生成两份缓存；编译失败不分类，语法错误被当成驱动问题无限重试；没有降级链，不支持的变体直接黑屏；去重按对象身份判断，相同内容反复编译；热更按文件全量失效，改一个文件把所有变体重编。shadervar/permute.py 是核心，tests/test_permute.py 15个红。要修的是：键名与布尔值归一、稳定且有上限的变体展开、包含源码/键/pipeline 版本的顺序无关缓存键、编译失败分类、按降级顺序回退、按内容去重、预算内批处理且不偷换变体、热更只失效受影响的键。

## 运行方式

```bash
python -m pytest tests -q
```

初始仓库输出恰好 `15 failed`：15 个测试全部命中 `NotImplementedError`，没有 collection / import / syntax error。仅使用标准库与 pytest，离线可跑，无随机、无 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_permute.py` 不可修改；实现只改 `shadervar/permute.py`（必要时可加 `shadervar/` 内的辅助模块）。
