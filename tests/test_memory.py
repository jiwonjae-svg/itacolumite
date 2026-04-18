"""Tests for memory system."""

from itacolumite.core.memory import ActionRecord, Memory


def test_memory_step_counter() -> None:
    mem = Memory()
    assert mem.step_count == 0
    assert mem.next_step() == 1
    assert mem.next_step() == 2
    assert mem.step_count == 2


def test_memory_short_term() -> None:
    mem = Memory(max_short_term=3)
    for i in range(5):
        record = ActionRecord(
            step=i + 1,
            timestamp="2024-01-01T00:00:00",
            action_type="mouse_click",
            params={"x": i * 10, "y": i * 10},
            observation="test",
            reasoning="test",
            confidence=0.5,
            result="success",
        )
        mem.record_action(record)

    recent = mem.get_recent_history(10)
    assert len(recent) == 3  # max_short_term=3
    assert recent[0].step == 3  # oldest remaining


def test_memory_task_lifecycle() -> None:
    mem = Memory()
    mem.start_task("test-001", "Test task")
    assert mem.current_task == "Test task"
    assert mem.task_id == "test-001"

    mem.next_step()
    mem.end_task("completed")
    assert mem.task_id == ""
    assert mem.current_task == ""


def test_history_summary_empty() -> None:
    mem = Memory()
    assert mem.get_history_summary() == "No actions taken yet."
