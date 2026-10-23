from processor.query_processor.nodes.g_node_answer_output import NodeAnswerOutput


def build_mock_state() -> dict:
    """构造模拟输入状态（与原 __main__ 中的 mock 数据一致）"""

    # 模拟重排序后的文档列表 (reranked_docs)
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2",
            "score": 0.95,
            "url": None,
            "content": """
                HAK 180 烫金机的操作面板位于机器正前方。
                开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
                具体的操作面板布局请参考下图：
                ![操作面板布局图](http://local-server/images/panel_view.jpg)

                如果是进行局部烫金，请调节侧面的旋钮。
                ![侧面旋钮细节](http://local-server/images/knob_detail.png)
                """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",
            "content": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "url": None,
            "content": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    return {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True,  # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }


def run_test() -> None:
    print("开始测试: 答案生成节点（外部调用）")

    # 1. 构造模拟数据
    mock_state = build_mock_state()

    # 2. 实例化并调用节点（通过 import 引入，而非在节点文件内运行）
    node_answer_output = NodeAnswerOutput()
    result = node_answer_output(mock_state)

    # 3. 验证 Prompt 构建
    if "prompt" in result:
        print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
    else:
        print("[FAIL] Prompt 未构建")

    # 4. 验证答案生成
    answer = result.get("answer")
    if answer and len(answer) > 10:
        print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
        print(f"答案预览: {answer[:50]}...")
    else:
        print(f"[WARN] 答案生成可能异常 (Content: {answer})")


if __name__ == "__main__":
    run_test()
