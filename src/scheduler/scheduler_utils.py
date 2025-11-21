"""
Scheduler Helper Functions
Timeline 출력 및 통계 유틸리티
"""

from typing import Dict, List, Set
from collections import defaultdict
from .event import EventType


class TimelineFormatter:
    """Timeline 출력 포맷팅 유틸리티"""
    
    @staticmethod
    def format_node_timeline(timeline: List, max_nodes: int = 50, show_tile_details: bool = True) -> None:
        """
        노드별 실행 Timeline 출력 (Transfer + Compute 구분, Tile 상세 출력)
        
        Args:
            timeline: Event timeline
            max_nodes: 출력할 최대 노드 수
            show_tile_details: Tile 단위 상세 출력 여부
        """
        if not timeline:
            return
        
        print("\n" + "=" * 80)
        print("⏱️  NODE EXECUTION TIMELINE")
        print("=" * 80)
        
        # 노드별 이벤트 수집
        node_timings = {}
        tile_data = {}  # {parent_node: {tile_id: {events, location}}}
        
        for event in timeline:
            if not event.node_id or event.node_id == "-":
                continue
            
            node_id = event.node_id
            event_type = event.event_type.value
            tag = None
            if hasattr(event, 'data') and event.data:
                tag = event.data.get('node_tag')
            transfer_parent = event.data.get('parent_node') if hasattr(event, 'data') and event.data else None
            if event.event_type in (EventType.TRANSFER_START, EventType.TRANSFER_DONE):
                target_id = transfer_parent or node_id
                TimelineFormatter._record_transfer_segment(
                    node_timings,
                    target_id,
                    event,
                    is_start=(event.event_type == EventType.TRANSFER_START)
                )
            
            # Tile 이벤트 처리 (개별 tile 추적)
            if hasattr(event, 'data') and event.data and 'parent_node' in event.data:
                parent_node = event.data['parent_node']
                
                # Parent node 초기화
                if parent_node not in node_timings:
                    node_timings[parent_node] = {'is_tiled': True, 'tiles': {}}
                if tag:
                    node_timings[parent_node]['tag'] = tag
                if parent_node not in tile_data:
                    tile_data[parent_node] = {}
                
                # Tile ID 추출 (node_id에서)
                tile_id = node_id
                
                if tile_id not in tile_data[parent_node]:
                    tile_data[parent_node][tile_id] = {
                        'events': {},
                        'array_id': event.data.get('array_id'),
                        'area_id': event.data.get('area_id')
                    }
                
                tile_data[parent_node][tile_id]['events'][event_type] = event.time_us
                
                # Parent node의 전체 시간 범위 업데이트
                if event_type in ['transfer_start', 'compute_start']:
                    if 'start_time' not in node_timings[parent_node]:
                        node_timings[parent_node]['start_time'] = event.time_us
                    else:
                        node_timings[parent_node]['start_time'] = min(
                            node_timings[parent_node]['start_time'], event.time_us
                        )
                elif event_type in ['transfer_done', 'compute_done']:
                    if 'end_time' not in node_timings[parent_node]:
                        node_timings[parent_node]['end_time'] = event.time_us
                    else:
                        node_timings[parent_node]['end_time'] = max(
                            node_timings[parent_node]['end_time'], event.time_us
                        )
                
                continue
            
            # 일반 노드 이벤트
            if node_id not in node_timings:
                node_timings[node_id] = {}
            
            node_timings[node_id][event_type] = event.time_us
            if tag:
                node_timings[node_id]['tag'] = tag
            
            # Location 정보 저장
            if hasattr(event, 'data') and event.data and 'location' not in node_timings[node_id]:
                if 'array_id' in event.data:
                    location_parts = [f"Arr{event.data['array_id']}"]
                    if 'area_id' in event.data:
                        location_parts.append(f"A{event.data['area_id']}")
                    node_timings[node_id]['location'] = ".".join(location_parts)
        
        # Tile data를 node_timings에 통합
        for parent_node, tiles in tile_data.items():
            node_timings[parent_node]['tiles'] = tiles
        
        # 출력
        TimelineFormatter._print_node_table_with_tiles(node_timings, max_nodes, show_tile_details)
    
    @staticmethod
    def _format_tile_locations(locs: Set) -> str:
        """Tile locations를 읽기 쉬운 문자열로 포맷"""
        array_areas = defaultdict(list)
        for arr_id, area_id in locs:
            array_areas[arr_id].append(area_id)
        
        location_strs = []
        for arr_id in sorted(array_areas.keys()):
            areas = sorted(array_areas[arr_id])
            if len(areas) == 1:
                location_strs.append(f"Arr{arr_id}.A{areas[0]}")
            elif len(areas) <= 3:
                area_str = ",".join(f"A{a}" for a in areas)
                location_strs.append(f"Arr{arr_id}.{area_str}")
            else:
                location_strs.append(f"Arr{arr_id}.A{areas[0]}-{areas[-1]}")
        
        return ", ".join(location_strs)
    
    @staticmethod
    def _print_node_table_with_tiles(node_timings: Dict, max_nodes: int, show_tile_details: bool) -> None:
        """노드 실행 정보 테이블 출력 (Tile 상세 포함)"""
        # 시작 시간 순으로 정렬
        def get_start_time(item):
            node_id, timings = item
            if 'start_time' in timings:
                return timings['start_time']
            return min(v for k, v in timings.items() if isinstance(v, (int, float)))
        
        sorted_nodes = sorted(node_timings.items(), key=get_start_time)
        
        # 출력할 노드 수 제한
        if max_nodes > 0 and len(sorted_nodes) > max_nodes:
            nodes_to_show = sorted_nodes[:max_nodes//2] + sorted_nodes[-max_nodes//2:]
            show_middle_omitted = True
        else:
            nodes_to_show = sorted_nodes
            show_middle_omitted = False
        
        # 헤더
        print(f"\n{'Node':50} | {'Start (us)':>10} | {'End (us)':>10} | {'Transfer':>10} | {'R/W (us)':>15} | {'Compute':>10} | {'Stall':>10} | {'Total':>10} | {'Location':10}")
        print("-" * 160)
        
        for i, (node_id, timings) in enumerate(nodes_to_show):
            if show_middle_omitted and i == max_nodes//2:
                print(f"{'...':^150}")
            
            # Tiled 노드인지 확인
            is_tiled = timings.get('is_tiled', False)
            
            if is_tiled and show_tile_details:
                # Tiled 노드: 전체 요약 + Tile 상세
                TimelineFormatter._print_tiled_node(node_id, timings)
            else:
                # 일반 노드
                transfer_summary = TimelineFormatter._summarize_transfer_segments(timings)
                start_time, end_time, transfer_time, compute_time, total_time = \
                    TimelineFormatter._calculate_node_times_simple(timings, transfer_summary)
                rw_column = TimelineFormatter._format_rw_column(transfer_summary['by_direction'])
                
                # Stall 시간 계산: Total - Transfer - Compute
                stall_time = max(0, total_time - transfer_time - compute_time)
                
                location = timings.get('location', '-')
                
                label = node_id[:50]
                tag = timings.get('tag')
                if tag:
                    label = f"{tag} {label}"
                print(f"{label:50} | {start_time:10.2f} | {end_time:10.2f} | {transfer_time:10.2f} | {rw_column:15} | {compute_time:10.2f} | {stall_time:10.2f} | {total_time:10.2f} | {location[:10]:10}")
        
        print("=" * 160)
        
        # 통계
        total_transfer = 0
        total_compute = 0
        total_transfer_by_direction = defaultdict(float)
        
        for _, timings in node_timings.items():
            transfer_summary = TimelineFormatter._summarize_transfer_segments(timings)
            total_transfer += transfer_summary['total']
            for direction, duration in transfer_summary['by_direction'].items():
                total_transfer_by_direction[direction] += duration
            if timings.get('is_tiled'):
                # Tiled 노드
                tiles = timings.get('tiles', {})
                for tile_id, tile_info in tiles.items():
                    events = tile_info['events']
                    if 'compute_done' in events and 'compute_start' in events:
                        total_compute += events['compute_done'] - events['compute_start']
            else:
                # 일반 노드
                if 'compute_done' in timings and 'compute_start' in timings:
                    total_compute += timings['compute_done'] - timings['compute_start']
        
        print(f"\n📊 Summary:")
        print(f"  Total Nodes: {len(node_timings)}")
        read_total = total_transfer_by_direction.get('read', 0.0)
        write_total = total_transfer_by_direction.get('write', 0.0)
        print(f"  Total Transfer Time: {total_transfer:.2f} us (Read {read_total:.2f} / Write {write_total:.2f})")
        print(f"  Total Compute Time: {total_compute:.2f} us")
        print("=" * 150)
    
    @staticmethod
    def _print_tiled_node(node_id: str, timings: Dict) -> None:
        """Tiled 노드 상세 출력"""
        tiles = timings.get('tiles', {})
        label = node_id
        if 'tag' in timings:
            label = f"{timings['tag']} {node_id}"
        
        if not tiles:
            return
        
        # 전체 노드 요약
        start_time = timings.get('start_time', 0)
        end_time = timings.get('end_time', 0)
        total_time = end_time - start_time
        
        # Tile 수 계산
        num_tiles = len(tiles)
        
        # Array별로 그룹화
        tiles_by_array = defaultdict(list)
        for tile_id, tile_info in tiles.items():
            array_id = tile_info.get('array_id', 0)
            tiles_by_array[array_id].append((tile_id, tile_info))
        
        # Location 문자열
        array_strs = []
        for arr_id in sorted(tiles_by_array.keys()):
            areas = set(tile_info.get('area_id', 0) for _, tile_info in tiles_by_array[arr_id])
            if len(areas) == 1:
                array_strs.append(f"Arr{arr_id}.A{list(areas)[0]}")
            else:
                array_strs.append(f"Arr{arr_id}.A{min(areas)}-{max(areas)}")
        location_str = ", ".join(array_strs)
        
        # 노드 전체 요약 (진하게 표시)
        header = f"🔸 {label[:48]}"
        print(f"{header:50} | {start_time:10.2f} | {end_time:10.2f} | {'':>10} | {'':>10} | {total_time:10.2f} | {location_str[:45]:45}")
        print(f"{'  └─ ' + str(num_tiles) + ' tiles':50} | {'':>10} | {'':>10} | {'':>10} | {'':>10} | {'':>10} | {'':45}")
        
        # 각 tile 상세
        for array_id in sorted(tiles_by_array.keys()):
            tile_list = tiles_by_array[array_id]
            
            for tile_id, tile_info in tile_list:
                events = tile_info['events']
                arr_id = tile_info.get('array_id', '-')
                area_id = tile_info.get('area_id', '-')
                
                # Tile 시간 계산
                transfer_start = events.get('transfer_start', 0)
                transfer_done = events.get('transfer_done', 0)
                compute_start = events.get('compute_start', 0)
                compute_done = events.get('compute_done', 0)
                
                tile_start = transfer_start if transfer_start > 0 else compute_start
                tile_end = compute_done
                transfer_time = transfer_done - transfer_start if transfer_start > 0 else 0
                compute_time = compute_done - compute_start if compute_start > 0 else 0
                tile_total = tile_end - tile_start
                
                # Tile 이름 짧게 (마지막 부분만)
                tile_short = tile_id.split('_')[-1] if '_' in tile_id else tile_id
                
                loc_str = f"Arr{arr_id}.A{area_id}"
                
                print(f"{'     ├─ ' + tile_short[:43]:50} | {tile_start:10.2f} | {tile_end:10.2f} | {transfer_time:10.2f} | {compute_time:10.2f} | {tile_total:10.2f} | {loc_str:45}")
    
    @staticmethod
    def _record_transfer_segment(node_timings: Dict, node_id: str, event, is_start: bool) -> None:
        """노드별 전송 구간 기록"""
        node_data = node_timings.setdefault(node_id, {})
        segments = node_data.setdefault('transfer_segments', {})
        event_data = event.data if hasattr(event, 'data') else None
        phase = (event_data.get('transfer_phase') if event_data else None) or 'shared_read'
        segment = segments.setdefault(phase, {})
        direction = (event_data.get('transfer_direction') if event_data else None) or segment.get('direction') or 'read'
        segment['direction'] = direction
        if event_data and event_data.get('location'):
            segment['location'] = event_data['location']
        key = 'start' if is_start else 'end'
        segment[key] = event.time_us
    
    @staticmethod
    def _summarize_transfer_segments(timings: Dict) -> Dict:
        """전송 구간 요약"""
        segments = timings.get('transfer_segments', {})
        summary = {'total': 0.0, 'by_direction': defaultdict(float)}
        for segment in segments.values():
            start = segment.get('start')
            end = segment.get('end')
            if start is None or end is None:
                continue
            duration = max(0.0, end - start)
            summary['total'] += duration
            direction = segment.get('direction', 'read')
            summary['by_direction'][direction] += duration
        return summary
    
    @staticmethod
    def _format_rw_column(direction_summary: Dict) -> str:
        """전송 방향별 시간을 문자열로 변환"""
        read_time = direction_summary.get('read', 0.0)
        write_time = direction_summary.get('write', 0.0)
        if read_time == 0 and write_time == 0:
            return "-"
        return f"R{read_time:.1f} / W{write_time:.1f}"
    
    @staticmethod
    def _calculate_node_times_simple(timings: Dict, transfer_summary: Dict = None) -> tuple:
        """일반 노드의 시작/종료/전송/연산 시간 계산"""
        transfer_summary = transfer_summary or TimelineFormatter._summarize_transfer_segments(timings)
        segments = timings.get('transfer_segments', {})
        
        # Shared_read가 있으면 그게 시작, 없으면 compute_start가 시작
        shared_read = segments.get('shared_read', {})
        read_start = shared_read.get('start', 0)
        
        transfer_ends = [seg.get('end') for seg in segments.values() if seg.get('end') is not None]
        last_transfer_end = max(transfer_ends) if transfer_ends else 0
        
        compute_start = timings.get('compute_start', 0)
        compute_done = timings.get('compute_done', 0)
        
        # 노드 시작 시간: shared_read가 있으면 그게 시작, 없으면 compute 시작
        if read_start > 0:
            start_time = read_start
        elif compute_start > 0:
            start_time = compute_start
        else:
            start_time = 0
        
        end_time = max(compute_done, last_transfer_end)
        transfer_time = transfer_summary['total']
        compute_time = compute_done - compute_start if compute_start > 0 else 0
        total_time = end_time - start_time if start_time > 0 else 0
        
        return start_time, end_time, transfer_time, compute_time, total_time
    
    @staticmethod
    def _calculate_node_times(timings: Dict) -> tuple:
        """노드의 시작/종료/전송/연산 시간 계산"""
        tiles = timings.get('tiles', [])
        
        if tiles:
            # Tiled 노드
            transfer_start = timings.get('transfer_start', 0)
            transfer_done = timings.get('transfer_done', 0)
            
            tile_starts = [t.get('compute_start', 0) for t in tiles if 'compute_start' in t]
            tile_dones = [t.get('compute_done', 0) for t in tiles if 'compute_done' in t]
            
            if tile_starts and tile_dones:
                compute_start = min(tile_starts)
                compute_done = max(tile_dones)
            else:
                compute_start = 0
                compute_done = 0
        else:
            # 일반 노드
            transfer_start = timings.get('transfer_start', 0)
            transfer_done = timings.get('transfer_done', 0)
            compute_start = timings.get('compute_start', 0)
            compute_done = timings.get('compute_done', 0)
        
        # 시간 계산
        if transfer_start > 0:
            start_time = transfer_start
        else:
            start_time = compute_start
        
        end_time = compute_done
        transfer_time = (transfer_done - transfer_start) if transfer_start > 0 else 0
        compute_time = compute_done - compute_start if compute_start > 0 else 0
        total_time = end_time - start_time
        
        return start_time, end_time, transfer_time, compute_time, total_time
    
    @staticmethod
    def print_overlap_summary(
        timeline: List,
        min_duration_us: float = 0.0,
        max_segments: int = 10,
        show_all_segments: bool = False
    ) -> None:
        """
        Transfer/Compute 중첩 구간 요약 출력
        
        Args:
            timeline: Event timeline
            min_duration_us: 이 시간 이상인 구간만 상세 출력
            max_segments: 상세 출력 최대 구간 수
        """
        if not timeline:
            return
        
        events = sorted(timeline, key=lambda e: e.time_us)
        active_compute = 0
        active_transfer = 0
        last_time = events[0].time_us
        segments = []
        
        for event in events:
            current_time = event.time_us
            if current_time > last_time:
                duration = current_time - last_time
                if (active_compute > 1 or
                        active_transfer > 1 or
                        (active_compute > 0 and active_transfer > 0)):
                    segments.append({
                        'start': last_time,
                        'end': current_time,
                        'duration': duration,
                        'compute': active_compute,
                        'transfer': active_transfer
                    })
            event_type = event.event_type
            if event_type == EventType.COMPUTE_START:
                active_compute += 1
            elif event_type == EventType.COMPUTE_DONE:
                active_compute = max(0, active_compute - 1)
            elif event_type == EventType.TRANSFER_START:
                active_transfer += 1
            elif event_type == EventType.TRANSFER_DONE:
                active_transfer = max(0, active_transfer - 1)
            last_time = current_time
        
        print("\n" + "=" * 80)
        print("🔁  PARALLEL / OVERLAP ANALYSIS")
        print("=" * 80)
        
        if not segments:
            print("중첩된 연산/전송 구간이 없습니다.")
            print("=" * 80)
            return
        
        total_overlap = sum(seg['duration'] for seg in segments)
        compute_parallel = sum(seg['duration'] for seg in segments if seg['compute'] > 1)
        transfer_parallel = sum(seg['duration'] for seg in segments if seg['transfer'] > 1)
        cross_overlap = sum(seg['duration'] for seg in segments if seg['compute'] > 0 and seg['transfer'] > 0)
        
        print(f"총 중첩 시간: {total_overlap:.2f} us")
        print(f"  - Compute 병렬: {compute_parallel:.2f} us")
        print(f"  - Transfer 병렬: {transfer_parallel:.2f} us")
        print(f"  - Compute↔Transfer 동시: {cross_overlap:.2f} us")
        
        # 시간 순으로 정렬된 구간만 표시
        detailed_segments = [seg for seg in segments if seg['duration'] >= min_duration_us]
        detailed_segments.sort(key=lambda s: s['start'])
        if max_segments > 0 and not show_all_segments:
            detailed_segments = detailed_segments[:max_segments]
        
        if detailed_segments:
            print("\n상세 구간 (시간 순):")
            for seg in detailed_segments:
                labels = []
                if seg['compute'] > 1:
                    labels.append(f"compute×{seg['compute']}")
                if seg['transfer'] > 1:
                    labels.append(f"transfer×{seg['transfer']}")
                if seg['compute'] > 0 and seg['transfer'] > 0:
                    labels.append("compute+transfer")
                label_str = ", ".join(labels) if labels else "-"
                print(f"  - {seg['start']:.2f}→{seg['end']:.2f} us "
                      f"({seg['duration']:.2f} us) [{label_str}]")
        else:
            print(f"\n길이 {min_duration_us} us 이상인 구간이 없습니다.")
        
        print("=" * 80)
    
    @staticmethod
    def format_memory_timeline(memory_events: List, max_events: int = 50, location_filter: str = None) -> None:
        """
        메모리 할당/해제 타임라인 출력
        
        Args:
            memory_events: 메모리 이벤트 리스트
            max_events: 출력할 최대 이벤트 수
            location_filter: 'shared', 'array', 또는 None
        """
        if not memory_events:
            return
        
        # Location 필터링
        if location_filter == 'shared':
            filtered_events = [e for e in memory_events if e['location'] == 'shared_sram']
            title = "📊 MEMORY ALLOCATION TIMELINE (Shared SRAM Only)"
        elif location_filter == 'array':
            filtered_events = [e for e in memory_events if e['location'].startswith('array_')]
            title = "📊 MEMORY ALLOCATION TIMELINE (Array SRAM Only)"
        else:
            filtered_events = memory_events
            title = "📊 MEMORY ALLOCATION TIMELINE (All Locations)"
        
        if not filtered_events:
            print(f"\n{title}")
            print("=" * 80)
            print("No events for this location filter.")
            print("=" * 80)
            return
        
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        
        # 시간순 정렬
        sorted_events = sorted(filtered_events, key=lambda x: x['time_us'])
        
        # 출력할 이벤트 수 제한
        if max_events > 0 and len(sorted_events) > max_events:
            events_to_show = sorted_events[:max_events//2] + sorted_events[-max_events//2:]
            show_middle_omitted = True
        else:
            events_to_show = sorted_events
            show_middle_omitted = False
        
        # 헤더
        print(f"\n{'Time (us)':>10} | {'Type':^6} | {'Location':^15} | {'Size (KB)':>10} | {'Buffer':40}")
        print("-" * 90)
        
        prev_location = None
        for i, event in enumerate(events_to_show):
            if show_middle_omitted and i == max_events//2:
                print(f"{'...':^90}")
            
            time_us = event['time_us']
            event_type = event['type']
            location = event['location']
            size_kb = event['size_kb']
            buffer = event['buffer']
            
            # 아이콘
            icon = "🟢" if event_type == 'alloc' else "🔴"
            type_str = "ALLOC" if event_type == 'alloc' else "FREE "
            
            # Location 그룹 구분
            if location != prev_location:
                if prev_location is not None:
                    print("-" * 90)
                prev_location = location
            
            print(f"{time_us:10.2f} | {icon} {type_str:4} | {location:15} | {size_kb:10.1f} | {buffer[:40]}")
        
        print("=" * 80)
        
        # 통계
        alloc_count = sum(1 for e in filtered_events if e['type'] == 'alloc')
        free_count = sum(1 for e in filtered_events if e['type'] == 'free')
        total_alloc_kb = sum(e['size_kb'] for e in filtered_events if e['type'] == 'alloc')
        
        print(f"\n📈 Statistics ({location_filter or 'all locations'}):")
        print(f"  Total Allocations: {alloc_count}")
        print(f"  Total Frees: {free_count}")
        print(f"  Total Allocated: {total_alloc_kb:.1f} KB")
        print("=" * 80)
