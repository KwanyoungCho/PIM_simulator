from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .activation import ActivationBuffer
from .event import Event, EventType


if TYPE_CHECKING:
    from .scheduler import InferenceScheduler


class ActivationPlanner:
    """
    Activation lifetime, 저장위치 결정, Shared write 전송 등을 담당하는 헬퍼.
    Scheduler의 상태/컨텍스트에 의존하며 복잡도를 격리한다.
    """

    def __init__(self, scheduler: "InferenceScheduler"):
        self.scheduler = scheduler

    # Convenience accessors -------------------------------------------------
    @property
    def ctx(self):
        return self.scheduler.ctx

    @property
    def activation_manager(self):
        return self.scheduler.activation_manager

    @property
    def pim(self):
        return self.scheduler.pim

    @property
    def memory_events(self):
        return self.scheduler.memory_events

    # Activation placement --------------------------------------------------
    def determine_locations(self, producer, consumers: List) -> List[str]:
        if not consumers:
            return ["shared_sram"]

        locations: List[str] = []
        producer_device = producer.device_type
        producer_id = producer.array_id if producer_device == "eflash" else producer.npu_id
        consumer_devices = {
            (node.device_type, node.array_id if node.device_type == "eflash" else node.npu_id)
            for node in consumers
        }

        has_cross_device = any(
            dev_type != producer_device or dev_id != producer_id for dev_type, dev_id in consumer_devices
        )
        if has_cross_device or len(consumers) > 1:
            locations.append("shared_sram")

        same_device_key = (producer_device, producer_id)
        if same_device_key in consumer_devices:
            if producer_device == "eflash":
                internal = f"array_{producer_id}_sram"
            else:
                internal = f"npu_{producer_id}_sram"
            if internal not in locations:
                locations.append(internal)

        if len(consumer_devices) == 1 and same_device_key in consumer_devices and len(consumers) == 1:
            return [locations[-1]] if locations else [f"array_{producer_id}_sram"]

        if not locations:
            locations.append("shared_sram")
        return locations

    def get_consumer_nodes(self, producer_node_id: str) -> List:
        consumers = []
        for node in self.scheduler.graph.get_all_nodes():
            if producer_node_id in node.input_nodes:
                consumers.append(node)
        return consumers

    def consumers_for_location(self, consumers: List, location: str, producer_array_id: Optional[int]) -> Set[str]:
        location_consumers: Set[str] = set()
        if location == "shared_sram":
            for consumer in consumers:
                if consumer.array_id != producer_array_id:
                    location_consumers.add(consumer.node_id)
        else:
            arr_id = int(location.split("_")[1])
            for consumer in consumers:
                if consumer.array_id == arr_id:
                    location_consumers.add(consumer.node_id)
        return location_consumers

    def optimal_location(self, buffer_id: str, consumer_array_id: int) -> str:
        lifetime = self.ctx.activation_lifetimes.get(buffer_id)
        if not lifetime:
            print(f"[Warning] No lifetime found for buffer {buffer_id}")
            return "shared_sram"
        storage_locations = lifetime["storage_locations"]
        internal = f"array_{consumer_array_id}_sram"
        if internal in storage_locations and storage_locations[internal]["ref_count"] > 0:
            return internal
        return "shared_sram"

    # Lifetime management ---------------------------------------------------
    def deallocate_activation(self, buffer_id: str, location: str):
        lifetime = self.ctx.activation_lifetimes.get(buffer_id)
        if not lifetime:
            return
        if location not in lifetime["storage_locations"]:
            return

        loc_info = lifetime["storage_locations"][location]
        freed_bytes = lifetime["buffer"].size_bytes

        self.memory_events.append(
            {
                "time_us": self.scheduler.current_time_us,
                "type": "free",
                "buffer": buffer_id,
                "location": location,
                "size_kb": freed_bytes / 1024,
            }
        )
        self.activation_manager.deallocate_from_sram(buffer_id, loc_info["sram"])
        del lifetime["storage_locations"][location]
        self.scheduler.deallocated_count += 1
        self.scheduler.deallocated_bytes += freed_bytes

    def record_array_copy(self, buffer_id: str, array_id: int, event_time_us: float):
        lifetime = self.ctx.activation_lifetimes.get(buffer_id)
        if not lifetime:
            return
        location = f"array_{array_id}_sram"
        if location in lifetime["storage_locations"]:
            return
        pending_consumers = lifetime["consumers"] - lifetime["used_by"]
        if not pending_consumers:
            return

        arr_consumers = {
            consumer_id
            for consumer_id in pending_consumers
            if (node := self.scheduler.graph.get_node(consumer_id)) and node.array_id == array_id
        }
        if not arr_consumers:
            return

        sram = self.pim.get_array(array_id).get_sram()
        if not sram.allocate(buffer_id, lifetime["buffer"].size_bytes, warn=True):
            return

        lifetime["storage_locations"][location] = {
            "sram": sram,
            "consumers": arr_consumers,
            "ref_count": len(arr_consumers),
        }
        self.memory_events.append(
            {
                "time_us": event_time_us,
                "type": "alloc",
                "buffer": buffer_id,
                "location": location,
                "size_kb": lifetime["buffer"].size_bytes / 1024,
                "consumers": list(arr_consumers),
            }
        )

    # Shared write orchestration --------------------------------------------
    def schedule_shared_write(
        self, node, buffer_id: str, activation_buffer: ActivationBuffer, event_time_us: float, node_tag: Optional[str]
    ):
        transfer_time = self.scheduler._calculate_transfer_time(activation_buffer.size_bytes)
        if transfer_time <= 0:
            return

        start_time = max(event_time_us, self.scheduler.shared_sram_bus_busy_until + self.scheduler.transfer_epsilon)
        done_time = start_time + transfer_time
        detail = {
            "buffer_id": buffer_id,
            "size_bytes": activation_buffer.size_bytes,
            "time_us": transfer_time,
            "direction": "write",
        }
        start_data = {
            "array_id": node.array_id,
            "area_id": node.area_id,
            "uses_shared_sram": True,
            "transfer_phase": "shared_write",
            "transfer_direction": "write",
            "buffer_id": buffer_id,
            "location": "shared_sram",
            "details": [detail],
        }
        if node_tag:
            start_data["node_tag"] = node_tag

        self.scheduler._add_event(
            Event(time_us=start_time, event_type=EventType.TRANSFER_START, node_id=node.node_id, data=start_data)
        )
        done_data = dict(start_data)
        done_data["transfer_time_us"] = transfer_time
        if node_tag:
            done_data["node_tag"] = node_tag
        self.scheduler._add_event(
            Event(time_us=done_time, event_type=EventType.TRANSFER_DONE, node_id=node.node_id, data=done_data)
        )
        self.scheduler.shared_sram_bus_busy_until = done_time

    def complete_shared_write(self, node, event: Event):
        buffer_id = event.data.get("buffer_id")
        if not buffer_id:
            return
        lifetime = self.ctx.activation_lifetimes.get(buffer_id)
        if not lifetime:
            return
        location = event.data.get("location", "shared_sram")
        loc_info = lifetime["storage_locations"].get(location)
        if not loc_info:
            return

        sram = loc_info.get("sram") or self.pim.get_shared_sram()
        if not loc_info.get("is_ready"):
            if not sram.allocate(buffer_id, lifetime["buffer"].size_bytes, warn=True):
                print(f"[Warning] Shared SRAM allocation failed for {buffer_id}")
                return
            loc_info["sram"] = sram
            loc_info["is_ready"] = True
            self.memory_events.append(
                {
                    "time_us": event.time_us,
                    "type": "alloc",
                    "buffer": buffer_id,
                    "location": location,
                    "size_kb": lifetime["buffer"].size_bytes / 1024,
                    "consumers": list(loc_info["consumers"]),
                }
            )
            
            # Shared write 완료 후, 이 버퍼를 기다리던 노드들을 다시 스케줄링
            self.scheduler._schedule_ready_nodes(event.time_us)

    # Completion -------------------------------------------------------------
    def finalize_node_completion(self, node, completion_time_us: float, schedule_ready: bool):
        node_id = node.node_id
        if node_id in self.scheduler.completed_nodes:
            return

        node.mark_done(completion_time_us)
        self.scheduler.completed_nodes.add(node_id)
        self.scheduler.graph.record_execution(node_id)
        if node_id in self.scheduler.running_nodes:
            del self.scheduler.running_nodes[node_id]
        if node_id in self.scheduler.running_npu_nodes:
            del self.scheduler.running_npu_nodes[node_id]
        if schedule_ready:
            self.scheduler._schedule_ready_nodes(completion_time_us)

